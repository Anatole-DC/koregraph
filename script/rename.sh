#!/bin/bash

sanitize() {
    echo "$1" | sed 's/\//\\\//g' | awk '{$1=$1};1'
}

usage() {
    echo "Usage: $0 -f <file> [-f <file2> ...] (--find <string> --replace-with <replacement_string> | --from <start_line> [--to <end_line>] --replace-with <replacement_string>)"
    exit 1
}

replace_with_find() {
    sanitized_find=$(sanitize "$find")
    sanitized_replace_with=$(sanitize "$replace_with")
    # echo "Trying to replace '${sanitized_find}' by '${sanitized_replace_with}'"
    sed -i "s/$sanitized_find/$sanitized_replace_with/g" "$file"
}

replace_with_range() {
    sanitized_replace_with=$(sanitize "$replace_with")
    if [ -z "$from" ]; then
        from=1
    fi
    if [ -z "$to" ]; then
        to='$'
    fi
    sed -i "${from},${to}c $sanitized_replace_with" "$file"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -f)
            shift
            files+=("$1")
            ;;
        --find)
            shift
            find="$1"
            ;;
        --replace-with)
            shift
            replace_with="$1"
            ;;
        --from)
            shift
            from="$1"
            ;;
        --to)
            shift
            to="$1"
            ;;
        *)
            usage
            ;;
    esac
    shift
done

if [ ${#files[@]} -eq 0 ] || [ -z "$replace_with" ]; then
    usage
fi

if [ -n "$find" ] && [ -n "$from" ]; then
    echo "Error: --find and --from cannot be used together."
    usage
fi

for file in "${files[@]}"; do
    if [ -n "$find" ]; then
        replace_with_find
    elif [ -n "$from" ] || [ -n "$to" ]; then
        replace_with_range
    else
        usage
    fi
done
