#!/bin/bash
# set -x

# List of api endpoints noted to be used in the script
# /rest/api/content/{page_id}/label'
# /rest/api/content/{page_id}?expand=body.storage'
# /rest/api/content"
# /rest/api/content"
# /rest/api/search"
# /rest/api/content/{content_id}/label"
# /rest/api/space"
# /rest/api/space/{space_key}/content/{content_type}"
# /rest/api/content/{item['id']}?expand=body.storage"
# /rest/api/user/current"
# /rest/api/space?next=true&limit={limit}&start={start}"
# /rest/api/space/{space_key}"
# /rest/api/content"
# /rest/api/content/{item_id}?expand=body.storage,space,version"
# /rest/api/space/{test_space}/permission"

API_ENDPOINTS=(
  "/rest/api/space"
  "/rest/api/space/{space_key}"
  "/rest/api/space/{space_key}/content/{content_type}"
  "/rest/api/content/{page_id}/label"
  "/rest/api/content/{page_id}?expand=body.storage"
  "/rest/api/content"
  "/rest/api/content"
  "/rest/api/search"
  "/rest/api/content/{content_id}/label"
  "/rest/api/space?next=true&limit={limit}&start={start}"
  "/rest/api/space/{space_key}"
  "/rest/api/content"
  "/rest/api/content/{item_id}?expand=body.storage,space,version"
  "/rest/api/space/{test_space}/permission"
  )

# Load .env file
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo ".env file not found!"
  exit 1
fi

# Check required variables
if [[ -z "$CONFLUENCE_DOMAIN" || -z "$USERNAME" || -z "$CONFLUENCE_ACCESS_TOKEN" || -z "$SPACES" ]]; then
  echo "Ensure CONFLUENCE_DOMAIN, USERNAME, SPACES, and CONFLUENCE_ACCESS_TOKEN are set in your .env file"
  exit 1
fi

# Split SPACES into an array
IFS=',' read -ra SPACES_ARRAY <<< "$SPACES"

# Initial URL
URL="$CONFLUENCE_DOMAIN/rest/api/space?next=true&limit=100&start=0"
AUTH_TOKEN="$(echo -n $USERNAME:$CONFLUENCE_ACCESS_TOKEN | base64)"
AUTH_TOKEN="$(echo -n $AUTH_TOKEN | tr -d '\n')"

ARRAY_OF_FOUND_SPACES=()

while [ -n "$URL" ]; do
  # remove the newlines from $AUTH_TOKEN
  response=$(curl -s -H "Authorization: Basic $AUTH_TOKEN" "$URL")

  # Print filtered spaces (alias in $SPACES_ARRAY)
  for alias in "${SPACES_ARRAY[@]}"; do
    # echo "$response" | jq -r '.results[] | select(.alias=="'$alias'") | {id, key, alias, name, type, status}'
    if [[ ! " ${ARRAY_OF_FOUND_SPACES[@]} " =~ " ${alias} " ]]; then
      ARRAY_OF_FOUND_SPACES+=("$alias")
    fi
  done

  # Determine next page URL
  next=$(echo "$response" | jq -r '._links.next // empty')

  if [ -n "$next" ]; then
    URL="$CONFLUENCE_DOMAIN$next"
  else
    URL=""
  fi
done

echo "Found spaces: ${ARRAY_OF_FOUND_SPACES[@]}"

declare -A URL_OF_PAGES
PAGE_IDS=()

# For each space in the array, grab the content and print the content
for space in "${ARRAY_OF_FOUND_SPACES[@]}"; do
  echo -e "\n---\nFetching content for space: $space\n---"
  # While next is not empty, get the next page for the page
  pages=$(curl -s -H "Authorization: Basic $AUTH_TOKEN" "$CONFLUENCE_DOMAIN/rest/api/space/$space/content" | jq -c '.page.results[] | {id, title, links_self: ._links.self}')
  next=$(curl -s -H "Authorization: Basic $AUTH_TOKEN" "$CONFLUENCE_DOMAIN/rest/api/space/$space/content" | jq -r '.page._links.next // empty')
  original_next=$next

  while [ -n "$next" ]; do
    # Paging through the pages without using a pipe that creates a subshell
    while IFS= read -r page; do
      page_id=$(echo "$page" | jq -r '.id')

      if [[ " ${PAGE_IDS[@]} " =~ " $page_id " ]]; then
        continue
      fi

      page_content_url=$(echo "$page" | jq -r '.links_self')
      
      if [[ ! "$page_id" =~ ^[0-9]+$ ]]; then
        echo "Page ID is not an integer: $page"
        exit 1
      fi

      if [[ ! "${URL_OF_PAGES[$space,$page_id]}" ]]; then
        PAGE_IDS+=("$page_id")
        URL_OF_PAGES[$space,$page_id]="$page_content_url"
        echo "URL_OF_PAGES[$space,$page_id]=$page_content_url"
      fi
    done <<< "$pages"
    
    echo "Length of PAGE_IDS: ${#PAGE_IDS[@]}"
    if [ -z "$next" ]; then
      continue
    fi
    pages=$(curl -s -H "Authorization: Basic $AUTH_TOKEN" "$CONFLUENCE_DOMAIN$next" | jq -c '.results[] | {id, title, links_self: ._links.self}')
    next=$(curl -s -H "Authorization: Basic $AUTH_TOKEN" "$CONFLUENCE_DOMAIN$next" | jq -r '.page._links.next // empty')
    if [ "$next" == "$original_next" ]; then
      echo "NEXT is the same as original_next: $next"
      continue
    fi
  done
done

# For each api endpoint, fetch the data and print the data
for endpoint in "${API_ENDPOINTS[@]}"; do
  echo -e "\n---\nFetching data from endpoint: ${endpoint}\n---"
  curl -s -H "Authorization: Basic $AUTH_TOKEN" "${CONFLUENCE_DOMAIN}${endpoint}"
done
