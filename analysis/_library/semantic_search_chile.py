#!/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Roy Gardner'
__copyright__ = 'Copyright 2023-2024, Roy and Sally Gardner'

"""
Code for search and clustering with tagging and export functionalities
"""

from packages import *
from utilities import *
import urllib
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import re
from openpyxl import Workbook
import os
from datetime import datetime

exclude_dict = {"segments": {}, "clusters": {}}

# =========================
# Helper Functions
# =========================

from scipy.spatial.distance import cosine, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

def cluster_search_results(results, model_dict, model_keys, cluster_threshold, draw_graph=False):
    """
    Cluster topic search segments by building a segment-segment similarity matrix.
    From the matrix, build a graph from linked pairs at or above a cluster threshold.
    Connected components in the graph are the clusters; unconnected singletons are the unclustered.
    
    Args:
        results (list): List of segment IDs and their metadata.
        model_dict (dict): Data model containing encoded segments and encodings.
        model_keys (dict): Keys for accessing necessary data in the model_dict.
        cluster_threshold (float): Minimum similarity value for linking two segments.
        draw_graph (bool): Whether to draw the similarity graph (default is False).
    
    Returns:
        dict: Dictionary of clusters, with integer keys for clusters and 'singletons' for unconnected segments.
    """
    # Extract indices for the relevant segments
    result_indices = [model_dict[model_keys['encoded']].index(v[0]) for v in results]
    result_encodings = [model_dict[model_keys['encodings']][i] for i in result_indices]

    n = len(result_encodings)
    if n == 0:
        return {"singletons": []}  # Handle case with no results

    # Create a similarity matrix
    matrix = np.zeros((n, n))
    row, col = np.triu_indices(n, 1)
    matrix[row, col] = pdist(result_encodings, lambda v, w: 1 - (np.arccos(1 - cosine(v, w)) / np.pi))

    # Threshold the matrix to create a binary graph adjacency matrix
    min_matrix = (matrix >= cluster_threshold).astype(np.int8)
    max_matrix = (matrix <= 1.0).astype(np.int8)
    t_matrix = np.bitwise_and(min_matrix, max_matrix)

    # Create a graph using the adjacency matrix
    graph = csr_matrix(t_matrix)
    _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # Collect the clusters and singletons
    component_dict = {}
    for i, label in enumerate(labels):
        if label in component_dict:
            component_dict[label].append(results[i])
        else:
            component_dict[label] = [results[i]]

    # Separate clusters and singletons
    cluster_dict = {label: component for label, component in component_dict.items() if len(component) > 1}

    # Reorder cluster IDs to ensure consistency
    label_list = list(cluster_dict.keys())
    cluster_dict = {label_list.index(label): component for label, component in cluster_dict.items()}

    # Add singletons
    singletons = [
        component[0] for label, component in component_dict.items() if len(component) == 1
    ]

    # Sort singletons by segment ID
    singletons = sorted(singletons, key=lambda x: parse_segment_id(x[0]))  # Use parse_segment_id for sorting

    # Add sorted singletons to the cluster dictionary
    if singletons:
        cluster_dict["singletons"] = singletons

    return cluster_dict

def handle_missing(value):
    """
    Converts missing values like -1 or None into a blank string.
    """
    return '' if value in [-1, None, ''] else value


def parse_segment_id(segment_id):
    """
    Parse a segment ID into a tuple for proper numeric sorting.
    Example: 'Sesión_100/Plenario/123/45' -> ('Sesión', 100, 'Plenario', 123, 45)
    """
    return tuple(int(part) if part.isdigit() else part for part in re.split(r'(\d+)', segment_id))

def run_search(choice_dict, model_dict, encoder, reviewed_segment_ids):
    """
    Run the search, enrich the results with metadata, cluster them,
    and display using the new cluster interface.
    """
    search_threshold = choice_dict["search_threshold"]
    cluster_threshold = choice_dict["cluster_threshold"]
    topic_text = choice_dict["topic"]

    # Encode the topic and calculate similarities
    encodings = encode_text([topic_text], encoder)
    encoding = [np.array(encodings).tolist()[0]]
    encoded_segments = model_dict["encoded_segments"]
    segment_encodings = model_dict["segment_encodings"]

    # Calculate similarity scores
    sim_list = cdist(encoding, segment_encodings, ad.angular_distance).flatten()
    results = [
        (encoded_segments[i], sim_list[i])
        for i in range(len(sim_list))
        if sim_list[i] >= search_threshold
    ]

    # Count before filtering
    total_results_before_filter = len(results)

    # Filter out reviewed segment IDs
    results = [r for r in results if r[0] not in reviewed_segment_ids]

    # Count after filtering
    total_results_after_filter = len(results)
    filtered_out_count = total_results_before_filter - total_results_after_filter

    # Logging
    print(f"Total results before filtering: {total_results_before_filter}")
    print(f"Number of segments filtered out: {filtered_out_count}")
    print(f"Total results after filtering: {total_results_after_filter}")

    # Add vertical space after the print statements
    spacer = widgets.HTML(
        value="<div style='margin-top: 20px;'></div>"  # Adjust the margin as needed
    )
    display(spacer)

    if not results:
        print("No results remaining after filtering.")
        return

    # Enrich results with metadata
    enriched_results = []
    for segment_id, score in results:
        document_id = "/".join(segment_id.split("/")[:-1])
        metadata = model_dict["documents_dict"].get(document_id, {})
        enriched_results.append((
            segment_id,
            handle_missing(model_dict["segments_dict"].get(segment_id, {}).get("text", "")),  # Text
            round(score, 2),  # Search score
            handle_missing(metadata.get("ID", "")),  # ID
            handle_missing(metadata.get("full_name", "Unknown")),  # Speaker
            handle_missing(metadata.get("district", "")),  # District
            handle_missing(metadata.get("region", "")),  # Region
            handle_missing(metadata.get("sex", "")),  # Sex
            handle_missing(metadata.get("age", "")),  # Age
            handle_missing(metadata.get("educ_level", "")),  # Education Level
            handle_missing(metadata.get("occupation", "")),  # Occupation
            handle_missing(metadata.get("elec_list", "")),  # Electoral List
            handle_missing(metadata.get("party", "")),  # Party
            handle_missing(metadata.get("collective", "")),  # Collective
            handle_missing(metadata.get("provisional_comm", "")),  # Provisional Commission
            handle_missing(metadata.get("thematic_comm", "")),  # Thematic Commission
            handle_missing(metadata.get("misc_comm", "")),  # Miscellaneous Commission
            handle_missing(metadata.get("ideology1", "")),  # Ideology score 1
            handle_missing(metadata.get("sd1", "")),  # Ideology score 1 SD
            handle_missing(metadata.get("ideology2", "")),  # Ideology score 2
            handle_missing(metadata.get("sd2", ""))  # Ideology score 2 SD
        ))

    # Cluster results
    cluster_dict = cluster_search_results(
        enriched_results,
        model_dict,
        {"encoded": "encoded_segments", "encodings": "segment_encodings"},
        cluster_threshold
    )

    # Display clusters with the updated interface
    display_clusters_with_context(cluster_dict, model_dict)


# =========================
# Tagging System
# =========================

tags_dict = {}  # Tracks tags for each segment

def add_tag(segment_id, new_tag):
    """
    Add a new tag to a segment in the tags_dict.
    """
    if segment_id not in tags_dict:
        tags_dict[segment_id] = []
    if new_tag not in tags_dict[segment_id]:  # Avoid duplicate tags
        tags_dict[segment_id].append(new_tag)

def remove_tag(segment_id, tag_to_remove):
    """
    Remove a specific tag from a segment in the tags_dict.
    """
    if segment_id in tags_dict and tag_to_remove in tags_dict[segment_id]:
        tags_dict[segment_id].remove(tag_to_remove)

def refresh_tags(segment_id, tag_display):
    """
    Refresh the displayed tags for a specific segment.
    """
    tag_display.value = f"Tags: {', '.join(tags_dict.get(segment_id, []))}"

def create_tagging_widgets(segment_id):
    """
    Create widgets for adding and managing tags for a segment.
    """
    # Text box for entering new tags
    tag_box = widgets.Text(
        placeholder='Enter a tag...',
        layout=widgets.Layout(width='200px')
    )
    
    # Buttons for adding and removing tags
    add_tag_button = widgets.Button(
        description="Add Tag",
        layout=widgets.Layout(width="80px")
    )
    remove_tag_button = widgets.Button(
        description="Remove Tag",
        layout=widgets.Layout(width="100px")
    )
    
    # Display area for the current tags
    tag_display = widgets.HTML(
        value=f"Tags: {', '.join(tags_dict.get(segment_id, []))}"
    )

    def on_add_tag_clicked(b):
        """
        Handle adding a new tag.
        """
        new_tag = tag_box.value.strip()
        if new_tag:
            add_tag(segment_id, new_tag)
            tag_box.value = ""  # Clear the input box
            refresh_tags(segment_id, tag_display)  # Update tag display

    def on_remove_tag_clicked(b):
        """
        Handle removing an existing tag.
        """
        tag_to_remove = tag_box.value.strip()
        if tag_to_remove:
            remove_tag(segment_id, tag_to_remove)
            tag_box.value = ""  # Clear the input box
            refresh_tags(segment_id, tag_display)  # Update tag display

    # Attach button click events to their respective handlers
    add_tag_button.on_click(on_add_tag_clicked)
    remove_tag_button.on_click(on_remove_tag_clicked)

    # Return the complete tagging widget
    return widgets.VBox([
        widgets.HBox([tag_box, add_tag_button, remove_tag_button]),
        tag_display
    ])


# =========================
# Context Functionality
# =========================

def get_context(segment_id, model_dict, context_window_value):
    """
    Retrieves the context segments for a given segment ID based on the current context window size.
    """
    all_segments = list(model_dict["segments_dict"].keys())
    idx = all_segments.index(segment_id)
    start = max(0, idx - context_window_value[0])  # Use the current value from the slider
    end = min(len(all_segments), idx + context_window_value[0] + 1)
    return [
        (
            context_id,
            model_dict["segments_dict"].get(context_id, {}).get("text", "No text found"),
            context_id == segment_id  # Flag the matching segment
        )
        for context_id in all_segments[start:end]
    ]


# =========================
# Main Display Functions
# =========================

def display_clusters_with_context(cluster_dict, model_dict):
    """
    Display search results organized by clusters with collapsible context windows and tagging functionality.
    Singletons will be listed last.
    """
    # Use a mutable `context_window_value` container to dynamically update the value
    context_window_value = [2]  # Start with a default value (e.g., 2)

    # Create a slider to adjust the context window dynamically
    context_slider = widgets.IntSlider(
        value=context_window_value[0],
        min=1,
        max=10,
        step=1,
        description="Context Window:",
        layout=widgets.Layout(width="400px"),
        style={'description_width': 'initial'}  # Ensure the description is fully visible
    )

    def update_context_window(change):
        """
        Updates the context window dynamically and re-renders the interface.
        """
        context_window_value[0] = change["new"]  # Update the current context window size

    context_slider.observe(update_context_window, names="value")

    def create_segment_widget(segment_data):
        """
        Creates a widget to display a single segment with its context, tagging functionality,
        and a "Do Not Export" checkbox.
        """
        segment_id = segment_data[0]
        segment_score = round(segment_data[2], 2)
        segment_text = segment_data[1]
        speaker = segment_data[4]

        # Button for Segment ID
        button = widgets.Button(
            description=segment_id,
            layout=widgets.Layout(width="auto"),
            style=dict(button_color="lightblue")
        )

        # "Do Not Export" checkbox
        exclusion_checkbox = widgets.Checkbox(
            value=exclude_dict["segments"].get(segment_id, False),  # Initialize based on exclude_dict
            description="Do not export segment",
            indent=False,
            layout=widgets.Layout(margin="0 0 0 10px")
        )

        # Update exclusion status in `exclude_dict` when the checkbox is toggled
        def update_exclusion(change):
            exclude_dict["segments"][segment_id] = change["new"]

        exclusion_checkbox.observe(update_exclusion, names="value")

        # Output widget for context
        context_output = widgets.Output()
        toggle_state = {"is_open": False}  # Track toggle state

        def toggle_context(b):
            """
            Toggles the display of the context window for a segment.
            """
            with context_output:
                if toggle_state["is_open"]:
                    toggle_state["is_open"] = False
                    clear_output()  # Hide context
                else:
                    toggle_state["is_open"] = True
                    clear_output()

                    # Add a horizontal line and display context
                    display(HTML("<hr>"))
                    display(HTML("<p><b>Context Window:</b></p>"))

                    # Add tagging widgets at the top of the context window
                    tagging_widgets = create_tagging_widgets(segment_id)
                    display(tagging_widgets)

                    # Retrieve and display context with metadata accordions
                    for idx, (ctx_id, ctx_text, is_match) in enumerate(
                        get_context(segment_id, model_dict, context_window_value)
                    ):
                        # Retrieve metadata for each segment
                        document_id = '/'.join(ctx_id.split('/')[:-1])  # Extract document ID
                        ctx_metadata = model_dict['documents_dict'].get(document_id, {})
                        ctx_speaker = handle_missing(ctx_metadata.get('full_name', 'Unknown'))

                        # Speaker and text
                        display(HTML(f"<p>{ctx_speaker}: {ctx_text}</p>"))

                        # Metadata accordion
                        context_metadata_content = widgets.VBox([
                            widgets.HTML(f"<p>ID: {handle_missing(ctx_metadata.get('ID', ''))}</p>"),
                            widgets.HTML(f"<p>Speaker: {ctx_speaker}</p>"),
                            widgets.HTML(f"<p>District: {handle_missing(ctx_metadata.get('district', ''))}</p>"),
                            widgets.HTML(f"<p>Region: {handle_missing(ctx_metadata.get('region', ''))}</p>"),
                            widgets.HTML(f"<p>Sex: {handle_missing(ctx_metadata.get('sex', ''))}</p>"),
                            widgets.HTML(f"<p>Age: {handle_missing(ctx_metadata.get('age', ''))}</p>"),
                            widgets.HTML(f"<p>Education: {handle_missing(ctx_metadata.get('educ_level', ''))}</p>"),
                            widgets.HTML(f"<p>Occupation: {handle_missing(ctx_metadata.get('occupation', ''))}</p>"),
                            widgets.HTML(f"<p>Electoral List: {handle_missing(ctx_metadata.get('elec_list', ''))}</p>"),
                            widgets.HTML(f"<p>Party: {handle_missing(ctx_metadata.get('party', ''))}</p>"),
                            widgets.HTML(f"<p>Collective: {handle_missing(ctx_metadata.get('collective', ''))}</p>"),
                            widgets.HTML(f"<p>Ideology 1: {handle_missing(ctx_metadata.get('ideology1', ''))}</p>"),
                            widgets.HTML(f"<p>Ideology 2: {handle_missing(ctx_metadata.get('ideology2', ''))}</p>")
                        ])
                        context_metadata_accordion = widgets.Accordion(children=[context_metadata_content])
                        context_metadata_accordion.set_title(0, f"Metadata for Segment {ctx_id}")
                        display(context_metadata_accordion)

                        # Add a horizontal line between context segments
                        if idx < len(get_context(segment_id, model_dict, context_window_value)) - 1:
                            display(HTML("<hr>"))

                    # Add a spacer and close button at the bottom
                    display(HTML("<div style='margin-top: 10px;'></div>"))
                    close_button = widgets.Button(
                        description="Close Context",
                        layout=widgets.Layout(width="120px")
                    )

                    def close_context(b):
                        """
                        Closes the context window when the button is clicked.
                        """
                        toggle_state["is_open"] = False
                        with context_output:
                            clear_output()  # Clear the context display

                    close_button.on_click(close_context)
                    display(close_button)

        button.on_click(toggle_context)

        # Combine the checkbox and button into a layout
        return widgets.VBox([
            widgets.HBox([button, exclusion_checkbox]),  # Add checkbox next to button
            widgets.HTML(f"<p><b>{speaker}:</b> {segment_text}</p>"),
            widgets.HTML(f"<b>Score:</b> {segment_score}"),
            context_output,
            widgets.HTML("<hr>")
        ])

    # Main display area for clusters
    display_area = widgets.VBox()
    # Main display area
    cluster_widgets = []

    for label, cluster in sorted(cluster_dict.items(), key=lambda x: x[0] == "singletons"):
        cluster_header = widgets.HTML(
            value=f"<h2 style='display: inline-block; margin-right: 10px;'><b>{'Singletons' if label == 'singletons' else f'Cluster {label}'}</b></h2>"
        )

        # "Do Not Export Cluster" checkbox
        cluster_exclusion_checkbox = widgets.Checkbox(
            value=exclude_dict["clusters"].get(label, False),
            description="Do not export cluster",
            indent=False,
            layout=widgets.Layout(height="24px")  # Ensure uniform height
        )
        
        def update_cluster_exclusion(change, cluster_label=label):
            exclude_dict["clusters"][cluster_label] = change["new"]
            for segment in cluster:
                exclude_dict["segments"][segment[0]] = change["new"]

        cluster_exclusion_checkbox.observe(update_cluster_exclusion, names="value")

        # Place the checkbox immediately next to the header
        cluster_widgets.append(
            widgets.VBox(  # Wrap header + checkbox and horizontal line in a VBox
                [
                    widgets.HBox(
                        [cluster_header, cluster_exclusion_checkbox],
                        layout=widgets.Layout(
                            align_items="center",  # Align both header and checkbox vertically
                            justify_content="flex-start",  # Keep them aligned to the left
                            gap="8px"  # Add space between the header and checkbox
                        )
                    ),
                    widgets.HTML(value="<hr>", layout=widgets.Layout(margin="-5px 0 10px 0"))  # Add horizontal line
                ]
            )
        )


        # Add segment widgets for the cluster
        cluster_widgets.extend([create_segment_widget(seg) for seg in cluster])

    # Export button
    export_button = widgets.Button(
        description="Export Tagged Results",
        layout=widgets.Layout(
            width="200px",
            margin="20px auto 10px auto",  # Center the button and add white space below
        ),
        style=dict(button_color="rgba(0, 255, 0, 0.2)")
    )

    def handle_export_button_click(b):
        export_with_tags_and_metadata(cluster_dict)

    export_button.on_click(handle_export_button_click)

    # Combine the slider, clusters, and export button
    display(widgets.VBox([context_slider, widgets.VBox(cluster_widgets), export_button]))


# =========================
# Export Functionality
# =========================

def export_with_tags_and_metadata(cluster_dict):
    """
    Export matching segments with their tags, semantic scores, and all metadata to an Excel file.
    """
    if not cluster_dict:
        print("No clusters to export.")
        return

    # Ensure the output folder exists
    output_folder = 'outputs/Chile'
    os.makedirs(output_folder, exist_ok=True)

    # Simplified file name based on the topic
    topic = choice_dict.get("topic", "no_topic").replace(" ", "_")  # Use topic, replace spaces with underscores

    # Construct the file name
    output_file = os.path.join(
        output_folder,
        f"{topic}.xlsx"
    )

    # Create a new workbook and select the active worksheet
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Clustered Results"

    # Write the header row, now including the "Semantic Score" column
    sheet.append([
        'Cluster', 'segment_ID', 'text', 'semantic_score', 'tags', 'ID', 'full_name',
        'district', 'region', 'sex', 'age_jul2021', 'educ_level', 'occupation',
        'elec_list', 'party', 'collective', 'provisional_commission', 'thematic_commission',
        'misc_commission', 'ideology1', 'sd1', 'ideology2', 'sd2'
    ])

    # Export clusters and singletons
    for cluster_label, segments in cluster_dict.items():
        if cluster_label == "singletons":
            print("Processing singletons...")
            for segment_data in segments:
                segment_id = segment_data[0]

                # Skip segments marked as "Do Not Export"
                if exclude_dict["segments"].get(segment_id, False):
                    continue

                # Append the segment to the sheet
                append_segment_to_sheet(sheet, "Singleton", segment_data)

        else:
            # Skip clusters marked as "Do Not Export"
            if exclude_dict["clusters"].get(cluster_label, False):
                print(f"Cluster {cluster_label} is excluded. Skipping...")
                continue

            cluster_name = f"Cluster {cluster_label}"

            for segment_data in segments:
                segment_id = segment_data[0]

                # Skip segments marked as "Do Not Export"
                if exclude_dict["segments"].get(segment_id, False):
                    continue

                # Append the segment to the sheet
                append_segment_to_sheet(sheet, cluster_name, segment_data)

    # Save the workbook to the specified file
    workbook.save(output_file)
    print(f"Export completed! File saved at: {output_file}")


def append_segment_to_sheet(sheet, cluster_name, segment_data):
    """
    Append a single segment's data, including semantic score, tags, and metadata, to the Excel sheet.
    """
    segment_id = segment_data[0]
    text = segment_data[1]
    semantic_score = segment_data[2]  # Include semantic score
    tags = ', '.join(tags_dict.get(segment_id, []))  # Get tags for the segment

    # Metadata fields (ensure indices match your enriched data structure)
    metadata = [
        segment_data[3],  # ID
        segment_data[4],  # Speaker (Full Name)
        segment_data[5],  # District
        segment_data[6],  # Region
        segment_data[7],  # Sex
        segment_data[8],  # Age
        segment_data[9],  # Education Level
        segment_data[10],  # Occupation
        segment_data[11],  # Electoral List
        segment_data[12],  # Party
        segment_data[13],  # Collective
        segment_data[14],  # Provisional Commission
        segment_data[15],  # Thematic Commission
        segment_data[16],  # Miscellaneous Commission
        segment_data[17],  # Ideology 1
        segment_data[18],  # Ideology 1 SD
        segment_data[19],  # Ideology 2
        segment_data[20],  # Ideology 2 SD
    ]

    # Combine cluster label, segment data, semantic score, tags, and metadata into a single row
    row = [cluster_name, segment_id, text, semantic_score, tags] + metadata
    sheet.append(row)
