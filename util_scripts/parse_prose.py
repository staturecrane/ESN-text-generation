"""
Removes newlines from ASCII file and joins into single text block to make it easier to split into sentences

:param file The location of the text file
: param output The path for the new text file
"""

import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="ASCII text file to parse into sentences")
parser.add_argument('-o', '--output', type=str, required=True, help='Path to write output file')

args = parser.parse_args()

def main(filename, output_filename):
    with open(os.path.abspath(filename)) as text_file:
        lines = text_file.read()
        # new_lines = map(lambda line: line.replace('\n', ' '), lines)
        stripped = filter(lambda line: line != ' ', lines)
        joined = ''.join(list(stripped))
        
        with open(os.path.abspath(output_filename), 'w') as newfile:
            newfile.write(joined)

if __name__ == '__main__':
    main(args.file, args.output)
