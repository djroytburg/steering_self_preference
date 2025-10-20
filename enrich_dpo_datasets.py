import json
import os.path as osp
import os
import glob

if __name__ == "__main__":
    jsons = []
    for file in glob.glob(osp.join("preference_extraction", "aware", "*.jsonl")):
        with open(file, "r") as f:
            for line in f:
                di = json.loads(line)
                di['file'] = file
                jsons.append(di)
    j = 0
    for item in open("train_dataset.jsonl","r").readlines():
        data = json.loads(item)
        your_response = data['prompt'].index("(your response)")
        other_response = data['prompt'].index("(other model's response)")
        self_first = your_response < other_response
        if self_first:
            self_summary = data['prompt'][your_response + len("your response):\n"):other_response - len('\nSummary2')].replace("\n","")
            for i, file in enumerate(jsons):
                found = False
                if self_summary in file['target_model_response']:
                    import json
                    import os.path as osp
                    import glob


                    def extract_between(prompt: str, start_marker: str, end_marker: str) -> str:
                        """Return the substring of prompt between start_marker and end_marker.
                        If either marker is missing, returns an empty string.
                        Trims surrounding whitespace and stray label text like 'Summary2'.
                        """
                        s = prompt.find(start_marker)
                        e = prompt.find(end_marker)
                        if s == -1 or e == -1:
                            return ""
                        # move past the start marker and any following separators like ':\n'
                        start = s + len(start_marker)
                        # strip common separators
                        while start < len(prompt) and prompt[start] in ": \n":
                            start += 1
                        substring = prompt[start:e].strip()
                        # remove a trailing 'Summary2' token if present
                        if substring.endswith('Summary2'):
                            substring = substring[: -len('Summary2')].strip()
                        return substring


                    if __name__ == "__main__":
                        # load jsonl records from the aware folder
                        jsons = []
                        aware_folder = osp.join("preference_extraction", "aware")
                        for path in glob.glob(osp.join(aware_folder, "*.jsonl")):
                            with open(path, "r") as f:
                                for line in f:
                                    di = json.loads(line)
                                    di["file"] = path
                                    jsons.append(di)

                        # read train dataset into memory once
                        train_path = "train_dataset.jsonl"
                        train_lines = []
                        with open(train_path, "r") as f:
                            train_lines = f.readlines()
                        total = len(train_lines)

                        j = 0
                        for item in train_lines:
                            data = json.loads(item)
                            prompt = data.get("prompt", "")
                            # find markers safely
                            your_marker = "(your response)"
                            other_marker = "(other model's response)"
                            your_idx = prompt.find(your_marker)
                            other_idx = prompt.find(other_marker)
                            if your_idx == -1 or other_idx == -1:
                                # skip malformed prompts but print a warning
                                print("WARN: markers not found in prompt; skipping")
                                continue

                            self_first = your_idx < other_idx
                            if self_first:
                                self_summary = extract_between(prompt, your_marker, other_marker)
                                found = False
                                for rec in jsons:
                                    if self_summary and self_summary in rec.get("target_model_response", ""):
                                        print(self_summary)
                                        print(rec.get("id"))
                                        print(rec.get("file"))
                                        if found:
                                            raise Exception("ANOMALOUS SELF: multiple aware records matched the same summary")
                                        found = True
                                        break
                                if found:
                                    j += 1
                            else:
                                other_summary = extract_between(prompt, other_marker, your_marker)
                                found = False
                                for rec in jsons:
                                    if other_summary and other_summary in rec.get("comparison_model_response", ""):
                                        print(other_summary)
                                        print(rec.get("id"))
                                        print(rec.get("file"))
                                        if found:
                                            raise Exception("ANOMALOUS OTHER: multiple aware records matched the same summary")
                                        found = True
                                        break
                                if found:
                                    j += 1

                            print(j, total)
                            self_pick = 1 if self_first else 2
                            type = "self_win" if data.get("chosen") == self_pick else "other_win"


