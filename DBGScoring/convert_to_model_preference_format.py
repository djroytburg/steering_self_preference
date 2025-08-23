#!/usr/bin/env python3
"""
Converter script to transform probability-based preference files into 
the model_preference format (equivalent to --use_infer_generate output).

Usage:
python convert_to_model_preference_format.py input_file.jsonl [input_file2.jsonl ...]

Example:
python convert_to_model_preference_format.py \
    model_preferences_fullset/xsum/evaluator_phi-4/gpt-3.5-turbo_llama3.1-8b-instruct.jsonl
"""

import json
import argparse
import os
import tempfile
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Convert probability-based preference files to model_preference format (in-place)')
    parser.add_argument('input_files', nargs='+', help='Input JSONL files with prob1/prob2 format')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be converted without modifying files')
    return parser.parse_args()


def convert_prob_to_model_preference(input_file, dry_run=False):
    """
    Convert probability-based preferences to model_preference format in-place.
    
    Args:
        input_file: Path to input JSONL file with prob1/prob2 format
        dry_run: If True, only show what would be converted without modifying
    """
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' does not exist")
        return False
    
    converted_count = 0
    skipped_count = 0
    tie_count = 0
    
    print(f"\n{'🔍 DRY RUN' if dry_run else '🔄 Converting'} {input_file}")
    
    # Read and convert data
    converted_data = []
    
    with open(input_file, 'r') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Check if already in model_preference format
                if 'model_preference' in data and 'prob1' not in data and 'prob2' not in data:
                    print(f"ℹ️  File already in model_preference format, skipping conversion")
                    return True
                
                # Extract required fields
                item_id = data.get('id')
                prob1 = data.get('prob1', 0)
                prob2 = data.get('prob2', 0)
                response1 = data.get('response1', '')
                response2 = data.get('response2', '')
                
                if item_id is None:
                    print(f"⚠️  Warning: Line {line_num} missing 'id' field, skipping")
                    skipped_count += 1
                    continue
                
                # Determine model preference based on probabilities with robust fallback
                try:
                    # Handle various edge cases
                    if prob1 is None or prob2 is None:
                        # Missing probability values - fallback to "1"
                        model_preference = "2"
                        tie_count += 1
                        if tie_count <= 3:
                            print(f"⚠️  Missing probability at line {line_num} (prob1={prob1}, prob2={prob2}), defaulting to '1'")
                    elif not isinstance(prob1, (int, float)) or not isinstance(prob2, (int, float)):
                        # Invalid probability types - fallback to "1"  
                        model_preference = "2"
                        tie_count += 1
                        if tie_count <= 3:
                            print(f"⚠️  Invalid probability types at line {line_num} (prob1={type(prob1)}, prob2={type(prob2)}), defaulting to '1'")
                    elif prob1 > prob2:
                        model_preference = "1"
                    elif prob2 > prob1:
                        model_preference = "2"
                    else:
                        # Tie (prob1 == prob2) - default to "1" 
                        model_preference = "2"
                        tie_count += 1
                        if tie_count <= 3:  # Only show first few ties
                            print(f"🤝 Tie at line {line_num} (prob1={prob1}, prob2={prob2}), defaulting to '1'")
                except Exception as e:
                    # Any other error in probability comparison - fallback to "1"
                    model_preference = "2"
                    tie_count += 1
                    if tie_count <= 3:
                        print(f"❌ Error comparing probabilities at line {line_num}: {e}, defaulting to '1'")
                
                # Create model_preference format
                converted_entry = {
                    'response1': response1,
                    'response2': response2,
                    'model_preference': model_preference,
                    'id': item_id
                }
                
                converted_data.append(converted_entry)
                converted_count += 1
                
            except json.JSONDecodeError:
                print(f"❌ Error: Line {line_num} is not valid JSON, skipping")
                skipped_count += 1
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
                skipped_count += 1
    
    if dry_run:
        print(f"\n📊 DRY RUN Results:")
        print(f"   ✅ Would convert: {converted_count} items")
        print(f"   🤝 Ties (defaulted to '1'): {tie_count} items")
        if skipped_count > 0:
            print(f"   ⚠️  Would skip: {skipped_count} items")
        print(f"\n📝 Sample converted entry:")
        if converted_data:
            print(json.dumps(converted_data[0], indent=2))
        return True
    
    # Write converted data back to file (in-place)
    try:
        # Use temporary file for safe in-place editing
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as temp_file:
            for entry in converted_data:
                temp_file.write(json.dumps(entry) + '\n')
            temp_file_path = temp_file.name
        
        # Replace original file with converted file
        shutil.move(temp_file_path, input_file)
        
        print(f"\n✅ Conversion complete!")
        print(f"   📊 Converted: {converted_count} items")
        print(f"   🤝 Ties (defaulted to '1'): {tie_count} items")
        if skipped_count > 0:
            print(f"   ⚠️  Skipped: {skipped_count} items")
        print(f"   📁 File updated in-place: {input_file}")
        
        # Show sample of converted format
        print(f"\n📝 Sample converted entry:")
        if converted_data:
            print(json.dumps(converted_data[0], indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        return False


def main():
    args = parse_args()
    
    success_count = 0
    total_files = len(args.input_files)
    
    print(f"🎯 Processing {total_files} file{'s' if total_files > 1 else ''}...")
    
    for input_file in args.input_files:
        if convert_prob_to_model_preference(input_file, args.dry_run):
            success_count += 1
    
    print(f"\n🏁 Final Results:")
    print(f"   ✅ Successfully processed: {success_count}/{total_files} files")
    
    if success_count < total_files:
        print(f"   ❌ Failed: {total_files - success_count} files")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
