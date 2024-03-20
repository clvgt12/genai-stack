#!/usr/bin/env python3
#
# A command line wrapper around the pdf_bot API
# The default API endpont URL is http://localhost:8504 unless otherwise specified
# Optionally, input can be supplied via a text file; the file name is specified on the command line with -f
# if no file is specified, then the script accepts input from stdin

import argparse
import requests, json
from urllib.parse import quote
import sys

def check_url(url):
    """Check if the URL is accessible."""
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

def write_output_to_file(chars, output_file):
    """output information to a file"""
    if output_file is not None:
        try:
            with open(output_file, 'a') as output_fp:
                print(chars, end='', flush=True, file=output_fp)
        except Exception as e:
            print(f"Error opening output file {output_file}: {e}")
            sys.exit(1)

def write_output(chars, output_file):
    """output information to stdout"""
    print(chars, end='', flush=True)
    if output_file is not None:
        write_output_to_file(chars, output_file)

def process_query(url, query, output_file):
    """Process a single query and reconstruct the message from tokenized data."""

    encoded_query = quote(query)
    full_url = f"{url}/query-stream?text={encoded_query}&rag=true&app_name=ask_your_docx"
    write_output(f">> ", output_file)
    
    try:
        response = requests.get(full_url, stream=True)
        for line in response.iter_lines():
            if line:  # Ensure the line is not empty
                line_str = line.decode('utf-8')  # Decode bytes to string
                # Attempt to find a JSON structure in the line
                if line_str.startswith('data: '):
                    try:
                        # Extract JSON string from line and parse it
                        json_str = line_str.split('data: ', 1)[1]
                        data = json.loads(json_str)
                        # Check if 'token' is present and add its value to the list
                        if 'token' in data:
                            write_output(data['token'], output_file)
                        continue
                    except json.JSONDecodeError:
                        # If there's an error decoding JSON, skip this line
                        continue
        write_output("\n", output_file)  # New line after processing
        return None
    except requests.RequestException as e:
        print(f"Error processing query: {e}")
        return None  # Return None or an appropriate response in case of error

def main(url, query_file=None, output_file=None):
    """Main function to process command-line arguments and run the script."""
    if not check_url(url):
        print(f"URL {url} is not accessible")
        sys.exit(1)

    # Open the output file in append mode if specified
    if output_file:
        write_output_to_file('', output_file)
    else:
        output_file = None

    if query_file:
        try:
            with open(query_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    write_output(f"> {line}\n", output_file)
                    process_query(url, line, output_file)
        except FileNotFoundError:
            print(f"File not found: {query_file}")
            sys.exit(1)
    else:
        while True:
            try:
                query = input("> ")
                if output_file is not None:
                    write_output_to_file(f"> {query}\n", output_file)        
                process_query(url, query, output_file)
            except EOFError as e:
                print("EOF")
                break
            except KeyboardInterrupt as e:
                print("ABORT")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line Interface for a REST API')
    parser.add_argument('-u', '--url', type=str, default='http://localhost:8504', help='URL of the API')
    parser.add_argument('-f', '--inputfile', type=str, help='Path to the input file')
    parser.add_argument('-o', '--outputfile', type=str, help='Path to the output file')
    args = parser.parse_args()
    main(args.url, args.inputfile, args.outputfile)
