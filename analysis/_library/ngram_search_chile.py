# =========================
# Imports and Setup
# =========================
import os
from openpyxl import Workbook
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import tempfile
import string
import json
import pandas as pd

# Global Variables and Setup
results = []  # Stores search results
results_with_context = {}  # Stores results with context
exclude_dict = {}  # Tracks exclusion status for segments
tags_dict = {}  # Tracks tags for each segment
searchable_segments = {}  # Preprocessed searchable segments
table = str.maketrans(dict.fromkeys(string.punctuation))

# Parameters
ngram = ''  # Default n-gram
context_window = 2  # Default context window
match_count_label = widgets.HTML(value=f"<b>Total Matches:</b> {len(results)}")

# =========================
# Preprocessing and Helper Functions
# =========================

# Support for n-gram search
for segment_id in model_dict['encoded_segments']:
    segment_text = model_dict['segments_dict'][segment_id]['text']
    clean_segment = segment_text.translate(table)  
    searchable_segments[segment_id] = ' ' + ' '.join((clean_segment.strip().lower()).split()) + ' '

def handle_missing(value):
    """
    Converts missing values like -1 or None into a blank string.
    """
    return '' if value in [-1, None, ''] else value

def parse_segment_id(segment_id):
    """
    Parse a segment ID into a tuple of integers and strings to ensure proper numeric sorting.
    Example: 'Sesion_66_Plenario_transcript/15/12' -> ('Sesion', 66, 'Plenario', 'transcript', 15, 12)
    """
    parts = []
    for part in segment_id.replace('_', '/').split('/'):  # Replace underscores with slashes for consistency
        if part.isdigit():  # Convert numeric parts to integers
            parts.append(int(part))
        else:  # Keep non-numeric parts as strings
            parts.append(part)
    return tuple(parts)
    
def bold_match(text, ngram):
    """
    Bolds the matched n-gram in the text.
    """
    return text.replace(ngram, f'<b>{ngram}</b>') if ngram else text

# Function to load reviewed segment IDs
def load_reviewed_segment_ids(outputs_folder="outputs/Chile"):
    reviewed_segment_ids = set()
    if not os.path.exists(outputs_folder):
        return reviewed_segment_ids  # Return empty set if folder doesn't exist

    for file_name in os.listdir(outputs_folder):
        if file_name.endswith(".xlsx"):
            file_path = os.path.join(outputs_folder, file_name)
            try:
                df = pd.read_excel(file_path)
                if 'segment_ID' in df.columns:
                    reviewed_segment_ids.update(df['segment_ID'].dropna().unique())
                else:
                    print(f"Warning: 'segment_ID' column not found in {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return reviewed_segment_ids

# =========================
# Core Logic
# =========================

# Function to dynamically update context window
def recalculate_context():
    """
    Recalculate the context for all matches based on the current context_window value.
    """
    global results_with_context
    results_with_context = {}
    for i, (segment_id, segment) in enumerate(sorted_segments):
        if segment_id in [result[0] for result in results]:
            start = max(i - context_window, 0)
            end = min(i + context_window + 1, len(sorted_segments))
            results_with_context[segment_id] = sorted_segments[start:end]

def perform_ngram_search():
    """
    Perform the n-gram search based on the current value of ngram,
    logging all information directly into match_count_label.
    """
    global results
    results = []

    # Load reviewed segments dynamically
    reviewed_segment_ids = load_reviewed_segment_ids()

    total_potential_matches = 0  # Total matches found before filtering
    filtered_segments_count = 0  # Matches that were filtered out

    if ngram:
        for i, (segment_id, segment) in enumerate(sorted_segments):
            if ngram.lower() in segment.get('text', '').lower():
                total_potential_matches += 1  # Count potential matches
                if segment_id in reviewed_segment_ids:
                    filtered_segments_count += 1  # Count as filtered
                    continue  # Skip already exported segments
                document_id = '/'.join(segment_id.split('/')[:-1])
                metadata = model_dict['documents_dict'].get(document_id, {})
                results.append((
                    segment_id,
                    handle_missing(segment.get('text', '')),
                    handle_missing(metadata.get('ID', '')),
                    handle_missing(metadata.get('full_name', '')),
                    handle_missing(metadata.get('district', '')),
                    handle_missing(metadata.get('region', '')),
                    handle_missing(metadata.get('sex', '')),
                    handle_missing(metadata.get('age', '')),
                    handle_missing(metadata.get('educ_level', '')),
                    handle_missing(metadata.get('occupation', '')),
                    handle_missing(metadata.get('elec_list', '')),
                    handle_missing(metadata.get('party', '')),
                    handle_missing(metadata.get('collective', '')),
                    handle_missing(metadata.get('provisional_comm', '')),
                    handle_missing(metadata.get('thematic_comm', '')),
                    handle_missing(metadata.get('misc_comm', '')),
                    handle_missing(metadata.get('position', '')),
                    handle_missing(metadata.get('ideology1', '')),
                    handle_missing(metadata.get('sd1', '')),
                    handle_missing(metadata.get('ideology2', '')),
                    handle_missing(metadata.get('sd2', ''))
                ))

    results.sort(key=lambda x: parse_segment_id(x[0]))

    recalculate_context()

    # Update match_count_label with all the log information
    match_count_label.value = (
        f"Total matches (pre-filtering): {total_potential_matches}<br>"
        f"Filtered out: {filtered_segments_count}<br>"
        f"Total matches (post-filtering): {len(results)}"
    )

# Initial calculation
sorted_segments = sorted(model_dict['segments_dict'].items(), key=lambda x: parse_segment_id(x[0]))
perform_ngram_search()

def export_with_tags_and_metadata():
    """
    Export matching segments with their tags and all metadata to an Excel file.
    """
    if not results:
        print("No results to export.")
        return  # Exit if there are no results

    # Ensure the output folder exists
    output_folder = 'outputs/Chile'
    os.makedirs(output_folder, exist_ok=True)

    # Name the output file
    output_file = os.path.join(output_folder, f'{ngram}_results.xlsx')

    # Create a new workbook and select the active worksheet
    workbook = Workbook()
    sheet = workbook.active

    # Write the header
    sheet.append([
        'segment_ID', 'text', 'tags', 'ID', 'full_name', 'district', 'region',
        'sex', 'age_jul2021', 'educ_level', 'occupation', 'elec_list', 'party',
        'collective', 'provisional_commission', 'thematic_commission', 'misc_commission', 
        'position','ideology1', 'sd1', 'ideology2', 'sd2'
    ])

    # Write the results
    for result in results:
        segment_id = result[0]
        if exclude_dict.get(segment_id, False):  # Skip excluded segments
            continue
        text = result[1]
        tags = ', '.join(tags_dict.get(segment_id, []))  # Get tags for the segment

        # Metadata fields
        metadata = [
            result[2],  # ID
            result[3],  # Speaker (Full Name)
            result[4],  # District
            result[5],  # Region
            result[6],  # Sex
            result[7],  # Age
            result[8],  # Education level
            result[9],  # Occupation
            result[10], # Electoral List
            result[11], # Party
            result[12], # Collective
            result[13], # Provisional Commission
            result[14], # Thematic Commission
            result[15], # Miscellaneous Commission
            result[16], # Position
            result[17], # Ideology 1
            result[18],  # Ideology 1 sd
            result[19], # Ideology 2
            result[20] # Ideology 2 sd
        ]

        # Combine all data into a single row
        row = [segment_id, text, tags] + metadata
        sheet.append(row)

    # Save the workbook to the output file
    workbook.save(output_file)

# =========================
# Widgets and UI Setup
# =========================

# Widgets

ngram_box = widgets.Text( 
    value=ngram,
    placeholder='Enter an n-gram...',
    description='N-gram:',
    disabled=False
)

set_ngram_button = widgets.Button(
    description="Set N-gram",
    layout=widgets.Layout(width="120px")
)

search_box = widgets.Text(
    placeholder='Type a segment ID or keyword...',
    description='Search:',
    disabled=False
)

clear_button = widgets.Button(
    description="Clear Search", 
    layout=widgets.Layout(width="120px")
)

slider_label = widgets.Label("Context Window:", layout=widgets.Layout(width='150px'))

context_slider = widgets.IntSlider(
    value=context_window,
    min=1,
    max=10,
    step=1,
    layout=widgets.Layout(width='200px')  # Adjust slider width
)

slider_with_label = widgets.HBox([slider_label, context_slider])  # Combine label and slider

export_button = widgets.Button(
    description="Export Tagged Results",
    layout=widgets.Layout(
        width="200px",
        margin="0 auto 10px auto",  # Center the button horizontally
        display="flex",
        justify_content="center"  # Align the button to the center
    ),
    style=dict(button_color="rgba(0, 255, 0, 0.2)")  # Transparent green
)

display_area = widgets.VBox()  # Dynamic display area for buttons and results


# Event Handlers

def update_ngram(b):
    """
    Updates the n-gram and refreshes the results when the button is clicked.
    """
    global ngram
    ngram = ngram_box.value.strip()  # Update the global n-gram
    perform_ngram_search()  # Refresh the results
    clear_output()  # Clear previous display
    display_interface()  # Redisplay the interface

def clear_search(b):
    """
    Clears the search box and resets the results.
    """
    search_box.value = ""  # Clear the search box

def update_context_window(change):
    """
    Updates the context window size dynamically based on slider input and refreshes the display.
    """
    global context_window
    context_window = change['new']
    recalculate_context()  # Recalculate context
    filter_results(None)  # Refresh the filtered display

def handle_export_button_click(b):
    """
    Handle the export button click by calling the export function.
    """
    export_with_tags_and_metadata()

# Attach handlers to widgets
set_ngram_button.on_click(update_ngram)
clear_button.on_click(clear_search)
export_button.on_click(handle_export_button_click)
context_slider.observe(update_context_window, names='value')

# =========================
# Context Display and Filtering
# =========================

def filter_results(change):
    """
    Filters results based on the search query and updates the display area.
    """
    search_query = search_box.value.lower()  # Get the search query
    filtered_buttons = []  # Initialize a list for filtered buttons

    # Loop through all results to check for matches
    for segment_id, text, _, speaker, *_ in results:
        # Check if the search query matches the segment_id or text
        if search_query in segment_id.lower() or search_query in text.lower():
            # Create the Segment ID button
            button = widgets.Button(
                description=segment_id,
                layout=widgets.Layout(
                    width="max-content",  # Allow button to resize dynamically
                ),
                style=dict(button_color="lightblue")  # Set the button's background color to light blue
            )

            # Add exclusion checkbox
            exclusion_checkbox = widgets.Checkbox(
                value=exclude_dict.get(segment_id, False),  # Get initial state
                description="Do not export",
                indent=False,
                layout=widgets.Layout(margin="0 0 0 10px")  # Add some spacing to the left
            )

            # Update exclusion status when the checkbox is toggled
            def update_exclusion(change, segment_id=segment_id):
                exclude_dict[segment_id] = change['new']

            exclusion_checkbox.observe(update_exclusion, names='value')

            # Combine the button and checkbox into an HBox
            button_with_checkbox = widgets.HBox([button, exclusion_checkbox])

            # Create the output widget for displaying context
            output_widget = output_widgets.get(segment_id, widgets.Output())  # Ensure output widget exists
            toggle_states[segment_id] = False  # Reset toggle state for filtered results

            # Attach the display_context function to the button
            button.on_click(display_context)

            # Add the combined button and checkbox, along with other info, to the filtered buttons
            filtered_buttons.append(
                widgets.VBox(
                    [
                        button_with_checkbox,
                        widgets.HTML(f"<b>{speaker}</b>: {bold_match(text, ngram)}"),
                        output_widget,
                        widgets.HTML("<hr>")  # Add horizontal line after each button
                    ],
                    layout=widgets.Layout(margin="10px 0px")  # Add vertical spacing between buttons
                )
            )

    # Update the display area with the filtered buttons
    display_area.children = filtered_buttons

search_box.observe(filter_results, names='value')

# Modify the display_context function to include tagging widgets
def display_context(button):
    segment_id = button.description
    document_id = '/'.join(segment_id.split('/')[:-1])  # Extract document ID
    output_widget = output_widgets[segment_id]
    toggle_states[segment_id] = not toggle_states[segment_id]  # Toggle the state

    with output_widget:
        clear_output()  # Clear previous output for the clicked button

        if toggle_states[segment_id]:  # If toggled on, display the context
            display(HTML("<hr>"))  # Horizontal line before context window
            display(HTML(f"<p><b>Context Window:</b></p>"))  # Not bolded
            display(HTML(f"<p>Selected Segment ID: {segment_id}</p>"))  # Not bolded

            # Add tagging widgets at the top of the context window
            tagging_widgets = create_tagging_widgets(segment_id)
            display(tagging_widgets)

            # Display context segments with their metadata in accordions
            for idx, (context_id, context_segment) in enumerate(results_with_context[segment_id]):
                context_document_id = '/'.join(context_id.split('/')[:-1])  # Extract document ID
                context_metadata = model_dict['documents_dict'].get(context_document_id, {})
                speaker = handle_missing(context_metadata.get('full_name', 'Unknown'))  # Get speaker name

                # Display the context segment text with speaker name first
                segment_text = bold_match(handle_missing(context_segment.get('text', 'No text found')), ngram)
                display(HTML(f"<p>{speaker}: {segment_text}</p>"))

                # Create metadata accordion for each context segment
                context_metadata_content = widgets.VBox([
                    widgets.HTML(f"<p>ID: {handle_missing(context_metadata.get('ID', ''))}</p>"),
                    widgets.HTML(f"<p>Speaker: {handle_missing(context_metadata.get('full_name', ''))}</p>"),
                    widgets.HTML(f"<p>District: {handle_missing(context_metadata.get('district', ''))}</p>"),
                    widgets.HTML(f"<p>Region: {handle_missing(context_metadata.get('region', ''))}</p>"),
                    widgets.HTML(f"<p>Sex: {handle_missing(context_metadata.get('sex', ''))}</p>"),
                    widgets.HTML(f"<p>Age: {handle_missing(context_metadata.get('age', ''))}</p>"),
                    widgets.HTML(f"<p>Education: {handle_missing(context_metadata.get('educ_level', ''))}</p>"),
                    widgets.HTML(f"<p>Occupation: {handle_missing(context_metadata.get('occupation', ''))}</p>"),
                    widgets.HTML(f"<p>Electoral List: {handle_missing(context_metadata.get('elec_list', ''))}</p>"),
                    widgets.HTML(f"<p>Party: {handle_missing(context_metadata.get('party', ''))}</p>"),
                    widgets.HTML(f"<p>Collective: {handle_missing(context_metadata.get('collective', ''))}</p>"),
                    widgets.HTML(f"<p>Provisional Commission: {handle_missing(context_metadata.get('provisional_comm', ''))}</p>"),
                    widgets.HTML(f"<p>Thematic Commission: {handle_missing(context_metadata.get('thematic_comm', ''))}</p>"),
                    widgets.HTML(f"<p>Miscellaneous Commission: {handle_missing(context_metadata.get('misc_comm', ''))}</p>"),
                    widgets.HTML(f"<p>Position: {handle_missing(context_metadata.get('position', ''))}</p>"),
                    widgets.HTML(f"<p>Ideology 1: {handle_missing(context_metadata.get('ideology1', ''))}</p>"),
                    widgets.HTML(f"<p>Ideology 2: {handle_missing(context_metadata.get('ideology2', ''))}</p>")
                ])
                context_metadata_accordion = widgets.Accordion(children=[context_metadata_content])

                # Set the accordion title as plain text (not bold)
                context_metadata_accordion.set_title(0, f"Metadata for Segment {context_id}")
                context_metadata_accordion.layout = widgets.Layout(font_weight="normal")  # Ensure no bold styling
                display(context_metadata_accordion)

                if idx < len(results_with_context[segment_id]) - 1:
                    display(HTML("<hr>"))  # Separate context segments

            # Add vertical space before the "Close Context" button
            spacer = widgets.HTML(value="<div style='margin-top: 10px;'></div>")
            display(spacer)

            # Add a "Close Context" button at the bottom
            close_button = widgets.Button(
                description="Close Context",
                layout=widgets.Layout(width="120px")
            )

            def close_context(b):
                """
                Close the context window when the button is clicked.
                """
                toggle_states[segment_id] = False  # Set state to closed
                with output_widget:
                    clear_output()  # Clear the context display

            close_button.on_click(close_context)  # Attach close functionality to the button
            display(close_button)

# =========================
# Tagging System
# =========================

# Add new tag to segment
def add_tag(segment_id, new_tag):
    """
    Add a new tag to a segment in the tags_dict.
    """
    if segment_id not in tags_dict:
        tags_dict[segment_id] = []
    if new_tag not in tags_dict[segment_id]:  # Avoid duplicate tags
        tags_dict[segment_id].append(new_tag)

# Remove tag from a segment
def remove_tag(segment_id, tag_to_remove):
    """
    Remove a specific tag from a segment in the tags_dict.
    """
    if segment_id in tags_dict and tag_to_remove in tags_dict[segment_id]:
        tags_dict[segment_id].remove(tag_to_remove)

def create_tagging_widgets(segment_id):
    """
    Create widgets for adding and managing tags for a segment.
    """
    tag_box = widgets.Text(
        placeholder='Enter a tag...',
        layout=widgets.Layout(width='200px')
    )
    add_tag_button = widgets.Button(description="Add Tag", layout=widgets.Layout(width="80px"))
    remove_tag_button = widgets.Button(description="Remove Tag", layout=widgets.Layout(width="100px"))
    tag_display = widgets.HTML(value=f"Tags: {', '.join(tags_dict.get(segment_id, []))}")

    def refresh_tags():
        """
        Refresh the displayed tags.
        """
        # Update the displayed list of tags
        tag_display.value = f"Tags: {', '.join(tags_dict.get(segment_id, []))}"

    def update_tags(b):
        """
        Add a new tag when the 'Add Tag' button is clicked.
        """
        new_tag = tag_box.value.strip()
        if new_tag:
            add_tag(segment_id, new_tag)
            tag_box.value = ""  # Clear the input box
            refresh_tags()  # Refresh tags display

    def remove_tag_action(b):
        """
        Remove a tag when the 'Remove Tag' button is clicked.
        """
        tag_to_remove = tag_box.value.strip()
        if tag_to_remove:
            remove_tag(segment_id, tag_to_remove)
            tag_box.value = ""  # Clear the input box
            refresh_tags()  # Refresh tags display

    # Attach actions to the buttons
    add_tag_button.on_click(update_tags)
    remove_tag_button.on_click(remove_tag_action)

    return widgets.VBox([
        widgets.HBox([tag_box, add_tag_button, remove_tag_button]),
        tag_display
    ])

# =========================
# Interface and Display
# =========================

# Global variables for UI state
buttons = []  # Stores dynamically created buttons for search results
output_widgets = {}  # Stores output widgets for each button
toggle_states = {}  # Tracks toggle states for each button

def display_interface():
    """
    Display the main interface with all widgets and buttons, ensuring Save/Load tools are next to N-Gram/Search tools.
    """
    global buttons
    buttons.clear()

    # Save/Load UI with consistent formatting
    save_session_box = widgets.Text(
        placeholder="Enter session name...",
        description="Save session:",
        layout=widgets.Layout(width="300px"),
        style={'description_width': 'initial'}  # Prevent the label from being cutoff
    )
    
    save_session_button = widgets.Button(
        description="Save",
        layout=widgets.Layout(width="80px"),
        style=dict(button_color="lightgreen")
    )
    
    session_dropdown = widgets.Dropdown(
        options=[],  # To be populated dynamically
        description="Load session:",
        layout=widgets.Layout(width="300px"),
        style={'description_width': 'initial'}  # Prevent the label from being cutoff
    )
    
    load_session_button = widgets.Button(
        description="Load",
        layout=widgets.Layout(width="80px"),
        style=dict(button_color="#FFA500")
    )

    # Align Save/Load tools vertically
    save_load_ui = widgets.VBox(
        [
            widgets.HBox([save_session_box, save_session_button]),
            widgets.HBox([session_dropdown, load_session_button]),
        ],
        layout=widgets.Layout(justify_content="flex-start", align_items="flex-start", margin="0 0 0 20px")
    )

    # Align N-Gram and Search tools vertically
    ngram_search_ui = widgets.VBox(
        [
            widgets.HBox([ngram_box, set_ngram_button]),
            widgets.HBox([search_box, clear_button]),
        ],
        layout=widgets.Layout(justify_content="flex-start", align_items="flex-start")
    )

    # Combine both tool sections in a single row, keeping them close
    tools_row = widgets.HBox(
        [ngram_search_ui, save_load_ui],
        layout=widgets.Layout(justify_content="flex-start", align_items="flex-start", width="100%")
    )

    # Save session functionality
    def save_session(b):
        session_name = save_session_box.value.strip()
        if not session_name:
            with export_feedback:
                export_feedback.clear_output()
                print("Error: Please enter a valid session name.")
            return

        os.makedirs("sessions", exist_ok=True)
        session_file = os.path.join("sessions", f"{session_name}.json")

        session_data = {
            "results": results,
            "tags_dict": tags_dict,
            "exclude_dict": exclude_dict
        }

        with open(session_file, "w") as f:
            json.dump(session_data, f)

        with export_feedback:
            export_feedback.clear_output()
            print(f"Session saved as: {session_file}")
        
        update_session_dropdown()  # Refresh dropdown options

    # Load session functionality
    def load_session(b):
        session_file = session_dropdown.value
        if not session_file:
            with export_feedback:
                export_feedback.clear_output()
                print("Error: Please select a session to load.")
            return

        try:
            with open(os.path.join("sessions", session_file), "r") as f:
                session_data = json.load(f)

            global results, tags_dict, exclude_dict, sorted_segments
            results = session_data.get("results", [])
            tags_dict = session_data.get("tags_dict", {})
            exclude_dict = session_data.get("exclude_dict", {})

            # Recalculate sorted_segments and context
            sorted_segments = sorted(model_dict["segments_dict"].items(), key=lambda x: parse_segment_id(x[0]))
            recalculate_context()  # Ensure results_with_context is updated based on loaded data

            with export_feedback:
                export_feedback.clear_output()

            clear_output()
            display_interface()

        except Exception as e:
            with export_feedback:
                export_feedback.clear_output()
                print(f"Error loading session: {str(e)}")

    save_session_button.on_click(save_session)
    load_session_button.on_click(load_session)

    # Update the dropdown menu with session files
    def update_session_dropdown():
        os.makedirs("sessions", exist_ok=True)
        session_files = [f for f in os.listdir("sessions") if f.endswith(".json")]
        session_dropdown.options = session_files

    update_session_dropdown()

    # Main content UI
    for segment_id, text, _, speaker, *_ in results:
        button = widgets.Button(
            description=segment_id,
            layout=widgets.Layout(
                width="auto",  # Dynamically adjust width
                overflow="visible"  # Prevent content clipping
            )
        )
        output_widget = widgets.Output()  # Create a unique output widget for each button
        toggle_states[segment_id] = False  # Initialize toggle state to False
        button.on_click(display_context)  # Attach the display function to the button
        output_widgets[segment_id] = output_widget  # Store the output widget for the segment
        buttons.append(
            widgets.VBox(
                [
                    button,
                    widgets.HTML(f"Text: {bold_match(text, ngram)}"),
                    widgets.HTML(f"Speaker: {speaker}"),
                    output_widget,
                    widgets.HTML("<hr>")  # Add horizontal line between buttons
                ],
                layout=widgets.Layout(margin="10px 0px")
            )
        )

    # Export feedback area
    export_feedback = widgets.Output()

    # Display everything
    display(widgets.VBox([
        tools_row,  # Aligned N-Gram/Search and Save/Load tools
        slider_with_label,
        match_count_label,
        display_area,
        export_button,
        export_feedback# Show feedback messages
    ]))
    filter_results(None)  # Initialize the filtered results

display_interface()