import json
import os.path as osp
import glob


def load_aware_records():
    """Load all records from preference_extraction/aware/*.jsonl"""
    jsons = []
    aware_folder = osp.join("preference_extraction", "aware")
    for path in glob.glob(osp.join(aware_folder, "*.jsonl")):
        with open(path, "r") as f:
            for line in f:
                di = json.loads(line)
                jsons.append(di)
    return jsons


if __name__ == "__main__":
    # Load aware records
    aware_records = load_aware_records()

    # Read test dataset
    input_file = "test_dataset_annotated.jsonl"
    output_file = "test_dataset_enriched.jsonl"

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            prompt = data.get("prompt", "")

            # Find markers
            your_marker = "(your response)"
            other_marker = "(other model's response)"
            your_idx = prompt.find(your_marker)
            other_idx = prompt.find(other_marker)

            if your_idx == -1 or other_idx == -1:
                print("WARN: markers not found in prompt; skipping")
                continue

            # Extract summaries
            self_marker = "(your response)"
            other_marker = "(other model's response)"
            self_start = prompt.find(self_marker)
            self_summary = ""
            other_summary = ""
            if self_start != -1:
                self_start += len(self_marker)
                while self_start < len(prompt) and prompt[self_start] in ":\n ":
                    self_start += 1
                # find end as \n\nSummary2 or other_marker
                end_marker = "\n\nSummary2"
                other_start = prompt.find(end_marker, self_start)
                if other_start == -1:
                    other_start = prompt.find(other_marker, self_start)
                if other_start != -1:
                    self_summary = prompt[self_start:other_start].strip()
                else:
                    self_summary = prompt[self_start:].strip()

                # now for other
                other_pos = prompt.find(other_marker)
                if other_pos != -1:
                    other_pos += len(other_marker)
                    while other_pos < len(prompt) and prompt[other_pos] in ":\n ":
                        other_pos += 1
                    other_summary = prompt[other_pos:].strip()
                    # Fix: if other_summary has extra text after \n\n, take only the first part
                    other_summary = other_summary.split('\n\n')[0].strip()

            # Find id where both summaries match the same record
            matching_id = None
            for rec in aware_records:
                target = rec.get("target_model_response", "")
                comp = rec.get("comparison_model_response", "")
                if self_summary == target and other_summary == comp:
                    matching_id = rec.get("id")
                    break

            # Add id to data if found
            if matching_id:
                data["id"] = matching_id

            # Write enriched data
            f_out.write(json.dumps(data) + "\n")
