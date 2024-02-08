#!/bin/bash
# A command line wrapper around the pdf_bot API
# Note: this script depends on perl and its JSON module to be installed

# _check_dependencies()
# Check external dependencies, exit with non zero status if there are failures
function _check_dependencies() {
    if ! command -v perl &> /dev/null; then
        echo "Perl is not installed."
        exit 1
    elif ! command -v curl &> /dev/null; then
        echo "curl is not installed."
        exit 1
    fi
    perl -MJSON -e 'exit' 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "perl's JSON module is installed."
        exit 1
    fi
}

# _check_url()
# Check to see if the url is accessible
function _check_url() {
    curl -o /dev/null -s -w "%{http_code}\n" "$1"
}

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

# _process_query()
# Process a single query
function _process_query() {
    local url="$1"
    local query="$2"; 
    local echo="$3"
    if [[ "$echo" = "Y" ]]; then echo "$query"; fi; echo -n '>> '
    local encoded_query=$(_urlencode "$query")
    curl -s "$url/query-stream?text=$encoded_query&rag=true" | \
        perl -MJSON -ne 'BEGIN{$|=1}if(/data: (.+)/){my $data=decode_json($1);print $data->{token} if exists $data->{token}}'
    perl -e 'print "\n\n"'
}

# main()
# Main function to process command-line arguments and run the script
function main() {
    local query_file=""
    local url="http://localhost:8504"
    while getopts "u:f:" opt; do
        case $opt in
            u) url="$OPTARG" ;;
            f) query_file="$OPTARG" ;;
            *) echo "Usage: $0 [-f query_file]"; exit 1 ;;
        esac
    done
    # Check dependencies
    _check_dependencies
    # Check if the url is accessible
    if [[ $(_check_url "$url") != "200" ]]; then
        echo "URL $url is not accessible"
        exit 1
    fi
    # Check if a file was specified
    if [[ -n $query_file ]]; then
        if [[ ! -f $query_file ]]; then
            echo "File not found: $query_file"
            exit 1
        fi
        # Read and process each line from the file
        while IFS= read -r line; do
            echo -n "> "
            _process_query "$url" "$line" "Y"
        done < "$query_file"
    else
        # Default behavior: read queries from standard input
        while true; do
            echo -n "> "
            if ! read query; then
                echo "EOF"
                break
            fi
            _process_query "$url" "$query" "N"
        done
    fi
}

main "$@"