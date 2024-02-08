# A command line wrapper around the pdf_bot API
# The default API endpont URL is http://localhost:8504 unless otherwise specified
# Optionally, input can be supplied via a text file; the file name is specified on the command line with -f
# if no file is specified, then the script accepts input from stdin

import argparse
import requests
import json
import sys
import urllib.parse

def check_module_availability():
    """Check if required modules are available."""
    required_modules = ['requests']
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    if missing_modules:
        print(f"Error: Missing required module(s): {', '.join(missing_modules)}")
        return False
    else:
        return True

def check_url(url):
    """Check if the URL is accessible."""
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

def process_query(url, query):
    """Process a single query."""
    encoded_query = urllib.parse.quote(query)
    full_url = f"{url}/query-stream?text={encoded_query}&rag=true"
    try:
        response = requests.get(full_url, stream=True)
        for line in response.iter_lines():
            print(f"{line}")
            if line.startswith(b'data: '):
                data = json.loads(line[6:])
                if 'token' in data:
                    print(data['token'])
                    break
    except requests.RequestException as e:
        print(f"Error processing query: {e}")

def main(url, query_file=None):
    """Main function to process command-line arguments and run the script."""
    if not check_module_availability():
        print(f"Required python modules are missing, exiting...")
        sys.exit(1)
    if not check_url(url):
        print(f"URL {url} is not accessible")
        sys.exit(1)

    if query_file:
        try:
            with open(query_file, 'r') as file:
                for line in file:
                    process_query(url, line.strip())
        except FileNotFoundError:
            print(f"File not found: {query_file}")
            sys.exit(1)
    else:
        while True:
            query = input("> ")
            if not query:
                print("EOF")
                break
            process_query(url, query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command Line Interface for a REST API')
    parser.add_argument('-u', '--url', type=str, default='http://localhost:8504', help='URL of the API')
    parser.add_argument('-f', '--file', type=str, help='Path to the query file')
    args = parser.parse_args()
    main(args.url, args.file)
