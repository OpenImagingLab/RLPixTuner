import os
import json
import argparse

def find_image_pairs(target_dir):
    # List to hold the image pairs
    image_pairs = []

    # Scan the target directory for .tif files
    for filename in os.listdir(target_dir):
        if filename.endswith('.tif'):
            # Extract the base name without the extension
            base_name = filename[:-4]
            if base_name.endswith('-Input'):
                target_name = base_name.replace('-Input', '-Target') + '.tif'
                if target_name in os.listdir(target_dir):
                    input_path = os.path.join(target_dir, filename)
                    target_path = os.path.join(target_dir, target_name)
                    image_pairs.append([input_path, target_path])

    # Write the image pairs to a JSON file
    output_file = os.path.join(target_dir, 'image_pairs.json')
    with open(output_file, 'w') as json_file:
        json.dump(image_pairs, json_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Generate image pairs from a directory containing .tif files.')
    parser.add_argument('directory', type=str, help='The directory to scan for .tif files')

    args = parser.parse_args()

    if os.path.isdir(args.directory):
        find_image_pairs(args.directory)
    else:
        print(f"Error: {args.directory} is not a valid directory.")

if __name__ == '__main__':
    main()