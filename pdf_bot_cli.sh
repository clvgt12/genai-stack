#!/bin/bash
# A command line wrapper around the pdf_bot API

# _urlencode()
# Encode a string as a safe URL
function _urlencode() {
    local length="${#1}"
    for (( i = 0; i < length; i++ )); do
        local c="${1:i:1}"
        case $c in
            [a-zA-Z0-9.~_-]) printf "$c" ;;
            *) printf '%%%02X' "'$c" ;;
        esac
    done
}

# _parse_response()
# Parse the passed JSON object, return the value of object with the key 'result'
function _parse_response() {
    perl -MJSON -0777 -e 'print ">> ",decode_json(<>)->{result},"\n\n"' <<< "$1"
}


# _process_query()
# Process a single query
function _process_query() {
    local query="$1"
    local encoded_query=$(_urlencode "$query")
    local response=$(curl -s "http://localhost:8504/query?text=$encoded_query&rag=true")
    echo "$query"
    _parse_response "$response"
}

# main()
# Main function to process command-line arguments and run the script
function main() {
    local query_file=""
    while getopts "f:" opt; do
        case $opt in
            f) query_file="$OPTARG" ;;
            *) echo "Usage: $0 [-f query_file]"; exit 1 ;;
        esac
    done
    # Check if a file was specified
    if [[ -n $query_file ]]; then
        if [[ ! -f $query_file ]]; then
            echo "File not found: $query_file"
            exit 1
        fi
        # Read and process each line from the file
        while IFS= read -r line; do
            echo -n "> "
            _process_query "$line"
        done < "$query_file"
    else
        # Default behavior: read queries from standard input
        while true; do
            echo -n "> "
            if ! read query; then
                echo "EOF"
                break
            fi
            _process_query "$query"
        done
    fi
}

main "$@"