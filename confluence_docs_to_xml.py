import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import re
import json
from urllib.parse import urljoin, urlparse

def scrape_confluence_rest_docs_to_xml(base_url):
    # Set up a session with headers that mimic a browser
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    })

    # Create the root XML element
    root = ET.Element("documentation")
    root.set("source", base_url)

    try:
        # Fetch the main page
        print(f"Fetching main page: {base_url}")
        response = session.get(base_url)
        response.raise_for_status()
        html_content = response.text
        
        # Print some debug info
        print(f"Main page status: {response.status_code}")
        print(f"Content type: {response.headers.get('Content-Type')}")
        print(f"Content length: {len(html_content)} bytes")
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Look for JavaScript variables that might contain API data
        # This is common in modern API docs where the data is loaded via JavaScript
        scripts = soup.find_all("script")
        api_data = None
        
        for script in scripts:
            script_text = script.string
            if not script_text:
                continue
                
            # Look for common patterns where API data is stored
            if "var data" in script_text or "apiData" in script_text or "resources" in script_text:
                print(f"Found potentially relevant script: {script_text[:50]}...")
                
                # Try to extract JSON data
                json_matches = re.findall(r'var\s+\w+\s*=\s*(\{[\s\S]*?\});', script_text)
                for json_match in json_matches:
                    try:
                        data = json.loads(json_match)
                        print(f"Successfully parsed JSON data with {len(str(data))} characters")
                        api_data = data
                        break
                    except json.JSONDecodeError:
                        continue
                        
            if api_data:
                break
                
        # Extract documentation from JavaScript data if found
        if api_data:
            print("Extracting API documentation from JavaScript data")
            extract_from_js_data(api_data, root)
        else:
            # No JS data found, try to extract directly from HTML
            print("No JavaScript data found, extracting directly from HTML")
            extract_from_html(soup, root, base_url)
            
        # Check if we have iframes that might contain content
        iframes = soup.find_all('iframe')
        if iframes and len(list(root)) < 5:  # If we haven't found much content yet
            print(f"Found {len(iframes)} iframes, checking for content")
            for iframe in iframes:
                iframe_src = iframe.get('src')
                if iframe_src:
                    iframe_url = urljoin(base_url, iframe_src)
                    try:
                        iframe_resp = session.get(iframe_url)
                        iframe_resp.raise_for_status()
                        iframe_soup = BeautifulSoup(iframe_resp.text, "html.parser")
                        extract_from_html(iframe_soup, root, iframe_url)
                    except Exception as e:
                        print(f"Error processing iframe {iframe_url}: {e}")
        
        # Check for any links to .html files that might be documentation pages
        print("Looking for HTML documentation pages")
        html_links = [a.get('href') for a in soup.find_all('a', href=True) 
                     if a.get('href').endswith('.html') or '/index.html#' in a.get('href')]
        
        unique_html_links = set()
        for link in html_links:
            if link.startswith('#'):  # Skip anchor links
                continue
            full_url = urljoin(base_url, link)
            # Extract the anchor part
            url_parts = full_url.split('#')
            base_page = url_parts[0]
            anchor = url_parts[1] if len(url_parts) > 1 else None
            
            if base_page not in unique_html_links:
                unique_html_links.add(base_page)
                try:
                    print(f"Fetching HTML page: {base_page}")
                    page_resp = session.get(base_page)
                    page_resp.raise_for_status()
                    page_soup = BeautifulSoup(page_resp.text, "html.parser")
                    
                    # If there's an anchor, try to find that specific section
                    if anchor:
                        print(f"Looking for anchor: #{anchor}")
                        # Try to find element with that id
                        anchor_elem = page_soup.find(id=anchor)
                        if anchor_elem:
                            # Create a new soup with just this element
                            section_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
                            section_soup.body.append(anchor_elem)
                            extract_from_html(section_soup, root, full_url)
                            continue
                    
                    # If no anchor or anchor not found, process the whole page
                    extract_from_html(page_soup, root, base_page)
                except Exception as e:
                    print(f"Error processing HTML page {base_page}: {e}")
        
        # Check if we got any content
        doc_items = list(root)
        print(f"Extracted {len(doc_items)} documentation items")
        
        if len(doc_items) == 0:
            # If we haven't found any content, add the raw HTML as a fallback
            print("No structured content found, adding raw HTML as fallback")
            doc_item = ET.SubElement(root, "docItem")
            doc_item.set("url", base_url)
            
            title_el = ET.SubElement(doc_item, "title")
            title_el.text = soup.title.string if soup.title else "Confluence REST API Documentation"
            
            content_el = ET.SubElement(doc_item, "content")
            content_el.text = html_content
            
            raw_el = ET.SubElement(doc_item, "rawHtml")
            raw_el.text = "true"
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        error_item = ET.SubElement(root, "error")
        error_item.text = str(e)
    
    # Return the XML as bytes
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

def extract_from_js_data(data, root):
    """Extract API documentation from JavaScript data object"""
    # The structure will depend on how the data is organized
    # This is a common pattern in API documentation sites
    if isinstance(data, dict):
        # Try to find resources, endpoints, or similar structures
        for key, value in data.items():
            if key in ['resources', 'endpoints', 'paths', 'api', 'methods']:
                if isinstance(value, dict):
                    for resource_name, resource_data in value.items():
                        add_resource(resource_name, resource_data, root)
                elif isinstance(value, list):
                    for i, resource_data in enumerate(value):
                        add_resource(f"Resource_{i}", resource_data, root)
            elif isinstance(value, (dict, list)):
                # Recursively search for nested API data
                extract_from_js_data(value, root)

def add_resource(name, data, root):
    """Add a resource entry to the XML"""
    doc_item = ET.SubElement(root, "docItem")
    
    title_el = ET.SubElement(doc_item, "title")
    title_el.text = name
    
    # Add description if available
    if isinstance(data, dict) and 'description' in data:
        desc_el = ET.SubElement(doc_item, "description")
        desc_el.text = data['description']
    
    # Add methods if available
    if isinstance(data, dict) and 'methods' in data and isinstance(data['methods'], (list, dict)):
        methods_el = ET.SubElement(doc_item, "methods")
        
        methods_data = data['methods']
        if isinstance(methods_data, dict):
            for method_name, method_data in methods_data.items():
                add_method(method_name, method_data, methods_el)
        elif isinstance(methods_data, list):
            for method_data in methods_data:
                method_name = method_data.get('name', 'Unknown')
                add_method(method_name, method_data, methods_el)

def add_method(name, data, parent_el):
    """Add a method entry to the XML"""
    method_el = ET.SubElement(parent_el, "method")
    method_el.set("name", name)
    
    # Add HTTP method if available
    if isinstance(data, dict):
        if 'httpMethod' in data:
            method_el.set("httpMethod", data['httpMethod'])
        elif 'method' in data:
            method_el.set("httpMethod", data['method'])
            
        # Add path if available
        if 'path' in data:
            path_el = ET.SubElement(method_el, "path")
            path_el.text = data['path']
            
        # Add description if available
        if 'description' in data:
            desc_el = ET.SubElement(method_el, "description")
            desc_el.text = data['description']
            
        # Add parameters if available
        if 'parameters' in data and isinstance(data['parameters'], (list, dict)):
            params_el = ET.SubElement(method_el, "parameters")
            params_data = data['parameters']
            
            if isinstance(params_data, dict):
                for param_name, param_data in params_data.items():
                    add_parameter(param_name, param_data, params_el)
            elif isinstance(params_data, list):
                for param_data in params_data:
                    param_name = param_data.get('name', 'Unknown')
                    add_parameter(param_name, param_data, params_el)

def add_parameter(name, data, parent_el):
    """Add a parameter entry to the XML"""
    param_el = ET.SubElement(parent_el, "parameter")
    param_el.set("name", name)
    
    if isinstance(data, dict):
        # Add type if available
        if 'type' in data:
            param_el.set("type", data['type'])
            
        # Add required flag if available
        if 'required' in data:
            param_el.set("required", str(data['required']).lower())
            
        # Add description if available
        if 'description' in data:
            desc_el = ET.SubElement(param_el, "description")
            desc_el.text = data['description']

def extract_from_html(soup, root, base_url):
    """Extract API documentation directly from HTML structure"""
    # Try to find API resource blocks
    resource_blocks = soup.select('.resource, .endpoint, .api-section, .rest-api, div[id^="resource_"]')
    
    if resource_blocks:
        print(f"Found {len(resource_blocks)} resource blocks in HTML")
        for block in resource_blocks:
            # Create a new doc item
            doc_item = ET.SubElement(root, "docItem")
            doc_item.set("url", base_url)
            
            # Try to find the resource name/title
            title_elem = block.find(['h1', 'h2', 'h3', 'h4', '.resource-title', '.title'])
            title_text = title_elem.get_text(strip=True) if title_elem else "Unknown Resource"
            
            title_el = ET.SubElement(doc_item, "title")
            title_el.text = title_text
            
            # Try to find the description
            desc_elem = block.select_one('.description, .summary, .notes, p')
            if desc_elem:
                desc_el = ET.SubElement(doc_item, "description")
                desc_el.text = desc_elem.get_text(strip=True)
            
            # Try to find methods
            method_blocks = block.select('.method, .operation, .http-method')
            
            if method_blocks:
                methods_el = ET.SubElement(doc_item, "methods")
                
                for method_block in method_blocks:
                    method_el = ET.SubElement(methods_el, "method")
                    
                    # Try to find method name
                    method_name_elem = method_block.select_one('.method-name, .operation-name, h3, h4')
                    method_name = method_name_elem.get_text(strip=True) if method_name_elem else "Unknown Method"
                    method_el.set("name", method_name)
                    
                    # Try to find HTTP method
                    http_method_elem = method_block.select_one('.http-method, .verb, .operation-method')
                    if http_method_elem:
                        http_method = http_method_elem.get_text(strip=True)
                        method_el.set("httpMethod", http_method)
                    
                    # Try to find path
                    path_elem = method_block.select_one('.path, .uri, .request-url')
                    if path_elem:
                        path_el = ET.SubElement(method_el, "path")
                        path_el.text = path_elem.get_text(strip=True)
                    
                    # Try to find method description
                    method_desc_elem = method_block.select_one('.method-description, .operation-description, .summary, p')
                    if method_desc_elem:
                        desc_el = ET.SubElement(method_el, "description")
                        desc_el.text = method_desc_elem.get_text(strip=True)
                    
                    # Try to find parameters
                    param_table = method_block.select_one('table.parameters, table.params, .params-table')
                    if param_table:
                        params_el = ET.SubElement(method_el, "parameters")
                        
                        param_rows = param_table.select('tr')
                        header_row = True
                        
                        for row in param_rows:
                            if header_row:
                                header_row = False
                                continue
                                
                            # Extract parameter data from table row
                            cells = row.select('td')
                            if len(cells) >= 1:
                                param_el = ET.SubElement(params_el, "parameter")
                                
                                # Name is usually in first column
                                param_el.set("name", cells[0].get_text(strip=True))
                                
                                # Type might be in second column
                                if len(cells) >= 2:
                                    param_el.set("type", cells[1].get_text(strip=True))
                                
                                # Description might be in third or last column
                                desc_index = min(2, len(cells) - 1)
                                if desc_index >= 0:
                                    desc_el = ET.SubElement(param_el, "description")
                                    desc_el.text = cells[desc_index].get_text(strip=True)
    else:
        # If we can't find specific resource blocks, try a more general approach
        print("No resource blocks found, trying general content extraction")
        
        # Try to find general content sections
        content_sections = soup.select('section, article, .content, .main, #content, .documentation')
        
        if content_sections:
            print(f"Found {len(content_sections)} general content sections")
            for section in content_sections:
                # Extract headings and their following content
                headings = section.find_all(['h1', 'h2', 'h3'])
                
                for heading in headings:
                    # Create a new doc item for each heading
                    doc_item = ET.SubElement(root, "docItem")
                    doc_item.set("url", base_url)
                    
                    # Use heading as title
                    title_el = ET.SubElement(doc_item, "title")
                    title_el.text = heading.get_text(strip=True)
                    
                    # Get all content until the next heading
                    content_parts = []
                    current = heading.next_sibling
                    
                    while current and not current.name in ['h1', 'h2', 'h3']:
                        if current.name and current.get_text(strip=True):
                            content_parts.append(current.get_text(strip=True))
                        current = current.next_sibling
                    
                    if content_parts:
                        content_el = ET.SubElement(doc_item, "content")
                        content_el.text = "\n".join(content_parts)
        else:
            # If still no content found, just grab all headings and paragraphs
            print("No content sections found, extracting all headings and paragraphs")
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
            
            for heading in headings:
                # Create a new doc item for each heading
                doc_item = ET.SubElement(root, "docItem")
                doc_item.set("url", base_url)
                
                # Use heading as title
                title_el = ET.SubElement(doc_item, "title")
                title_el.text = heading.get_text(strip=True)
                
                # Get the next paragraph if available
                next_p = heading.find_next('p')
                if next_p:
                    content_el = ET.SubElement(doc_item, "content")
                    content_el.text = next_p.get_text(strip=True)

if __name__ == "__main__":
    base_url = "https://docs.atlassian.com/atlassian-confluence/REST/6.6.0/"
    print(f"Starting scrape of {base_url}")
    xml_data = scrape_confluence_rest_docs_to_xml(base_url)
    
    # Write to file
    with open("confluence_rest_docs.xml", "wb") as f:
        f.write(xml_data)
    
    print(f"Scraping complete. XML saved to confluence_rest_docs.xml")
    
    # Print a preview of the XML
    xml_preview = xml_data.decode('utf-8')[:1000] + "..." if len(xml_data) > 1000 else xml_data.decode('utf-8')
    print(f"\nXML Preview:\n{xml_preview}")