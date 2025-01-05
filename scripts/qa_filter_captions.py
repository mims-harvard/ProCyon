import os, time, argparse, math
from tqdm import trange
import torch
import pandas as pd
import numpy as np
import pickle

from procyon.model.model_unified import UnifiedProCyon
from procyon.data.inference_utils import create_caption_input_simple, create_qa_input_simple, uniprot_id_to_index, index_to_uniprot_id, ProCyonQAInference

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_args, model_args, train_args = UnifiedProCyon.get_checkpoint_configs(resume_from_checkpoint = args.ckpt)
    # Load model:
    model, _ = UnifiedProCyon.from_pretrained(checkpoint_dir=args.ckpt)
    model.to(device)
    model.eval()

    qa_model = ProCyonQAInference(model, device = device)

    # Gather captions in directory:
    if args.caption_fpath is not None:
        if args.caption_fpath.endswith("tsv.gz"):
            caption_df = pd.read_csv(args.caption_fpath, compression="gzip", sep="\t")
        elif args.caption_fpath.endswith(".pickle") or args.caption_fpath.endswith(".pkl"):
            caption_df = pd.read_pickle(args.caption_fpath)
        elif args.caption_fpath.endswith(".csv"):
            caption_df = pd.read_csv(args.caption_fpath)

    else:
        all_df_list = []
        for f in os.listdir(args.caption_dir):
            df = pd.read_pickle(os.path.join(args.caption_dir, f))
            all_df_list.append(df)

        caption_df = pd.concat(all_df_list)

    # Calculate chunks:
    if (args.num_chunks is not None) and (args.chunk_idx is not None):
        fs = caption_df.shape[0]
        chunk_indices = np.arange(0, fs, math.ceil(fs / args.num_chunks))
        chunk_start = chunk_indices[args.chunk_idx]

        if args.chunk_idx == (args.num_chunks - 1):
            chunk_end = fs

        else:
            chunk_end = chunk_indices[args.chunk_idx + 1]

        # Split by this job's chunk size:
        caption_df = caption_df.iloc[chunk_start:chunk_end,:]

    all_scores_dict = {
        "uniprot_id": [],
        "response_num": [],
        "caption_output": [],
        "yes": [],
        "no": []
    }

    # Iterate over QA passes
    for i in trange(caption_df.shape[0]):
        
        query_row = caption_df.iloc[i,:]
        query_uniprot_id = query_row["uniprot_id"]

        response_levels = sorted([c for c in query_row.index if "response" in c])

        # Iterate over responses:
        for r in response_levels:
            query_caption = query_row[r]

            # Create QA input:
            input_qa_simple = create_qa_input_simple(
                input_aaseq_ids = [uniprot_id_to_index(query_uniprot_id)],
                data_args = data_args,
                input_description = query_caption,
                drug_inputs = None,
                task_definition = None,
                instruction_source_dataset = args.prompt_dataset,
                instruction_source_relation = args.prompt_relation,
                aaseq_type = "protein",
                icl_example_number = 1,
                device = device,
            )

            # Run QA fwd pass:
            with torch.no_grad():
                model_qa_out = qa_model(input_qa_simple)

            # Deconstruct output:
            yes_score = model_qa_out['pred'][0,qa_model.yes_token].item()
            no_score = model_qa_out['pred'][0,qa_model.no_token].item()

            # Add all to dict:
            all_scores_dict["uniprot_id"].append(query_uniprot_id)
            all_scores_dict["response_num"].append(r)
            all_scores_dict["caption_output"].append(query_caption)
            all_scores_dict["yes"].append(yes_score)
            all_scores_dict["no"].append(no_score)


    df = pd.DataFrame(all_scores_dict)
    print(df)
    if args.save_path is not None:
        df.to_csv(args.save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required = True, help = "Path to ProCyon checkpoint")
    parser.add_argument("--caption_dir", required = False, type = str, help = "Directory with captions")
    parser.add_argument("--caption_fpath", required = False, type = str, help = "File with captions to filter. Should be in the style of caption_bulk.py output.")
    parser.add_argument("--save_path", required = False, default = None, type = str, help = "CSV file to save to")

    parser.add_argument("--chunk_idx", required = False, default = None, type = int, help = "Used for chunking the dataframe into separate pieces")
    parser.add_argument("--num_chunks", required = False, default = None, type = int, help = "Used for chunking the dataframe into separate pieces")

    parser.add_argument("--prompt_dataset", default = "uniprot", required = False, 
        help = "Dataset prompt to use for the model. Should match the one used for generating captions.")
    parser.add_argument("--prompt_relation", default = "all", required = False,
        help = "Relation for the dataset. Should match the one used for generating captions.")

    args = parser.parse_args()

    assert (args.caption_fpath is not None) or (args.caption_dir is not None)

    main(args)