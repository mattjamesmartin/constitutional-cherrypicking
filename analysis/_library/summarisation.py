#!/bin/python
# -*- coding: utf-8 -*-

__author__      = 'Roy Gardner'
__copyright__   = 'Copyright 2022-2024, Roy and Sally Gardner'

"""

Code for graph-based clustering for summarisation

"""
from packages import *
from utilities import *
import urllib

def clean_name(name):
    name = ' '.join(name.split('\r\n'))
    name = ' '.join(name.split('\n\r'))
    name = ' '.join(name.split('\n'))
    name = ' '.join(name.split('\r'))
    return name.strip()


def get_doc_ids(segment_ids,model_dict,model_keys):
    return [sid.split('/')[0] for sid in segment_ids]

def get_summarisation(matrix,model_dict,threshold=0.8):
    """
    Generate graph-based summarisation topics
    param summarisation_choices: user interface choices
    param matrix: segments matrix we are clustering
    param model_dict: Model dictionary
    return: Dictionary of summarisation topics (components) where key is topic ID, value is list of
    tuples in form (segment_id,centrality of segment)
    """
    matrix = np.triu(matrix,k=1)
    min_matrix = (matrix >= 0.7).astype(np.int8)
    max_matrix = (matrix <= 1.0).astype(np.int8)
    t_matrix = np.bitwise_and(min_matrix,max_matrix)
    graph = csr_matrix(t_matrix)
    _,labels = connected_components(csgraph=graph, directed=False,return_labels=True)
    # Collect the components
    # If there is a segment indices list in the model_dict, then the segments_matrix is a subset and we need
    # to use the segments list to get the correct segment IDs
    summarisation_dict = {}
    segment_indices = range(0,matrix.shape[0])
    for i,label in enumerate(labels):
        segment_id = model_dict['encoded_segments'][segment_indices[i]]

        if label in summarisation_dict:
            summarisation_dict[label].append((segment_id,len(graph[i].indices)))
        else:
            summarisation_dict[label] = [(segment_id,len(graph[i].indices))]
    summarisation_dict = {label:component for label,component in summarisation_dict.items() if len(component) > 1}

    return summarisation_dict

def convert_date(date):
    # This is bad
    if type(date) == str:
        return int(''.join(date.split('-')))
    else:
        return date

def get_summarisation_matrix(summarisation_dict,model_dict,config):
    """
    
    """
    matrix = []

    name_field = config['document_metadata']['name'][1]
    date_field = config['document_metadata']['date'][1]

    doc_data = [(k,v[name_field].strip(),convert_date(v[date_field])) for k,v in model_dict[config['model_keys']['documents']].items()]
    doc_data = sorted(doc_data,key=lambda t:t[2])
    doc_ids = [t[0] for t in doc_data]
    for _,component in summarisation_dict.items():
        component_doc_ids = [t[0].split('/')[0] for t in component]
        row = [0]*(len(doc_ids))
        for i,doc_id in enumerate(doc_ids):
            # Does the document contain a section that belongs to the topic component
            doc_segment_count = len([v for v in component_doc_ids if v==doc_id])
            if doc_segment_count > 0:
                row[i] = 1
            else:
                row[i] = 0
        matrix.append(row)

    matrix = np.array(matrix)
    # Get order of topic appearance in the source sequence by
    # finding the first non-zero value in a summarisation topic row in the matrix
    row_indices = []
    for i,row in enumerate(matrix):
        # Tuple in form (row_index,col_index) where col_index defines first appearance
        row_indices.append((i,np.argwhere(row)[0][0]))
    # Sort by the col_index part of the tuple so we can reorder matrix rows
    row_indices = [t[0] for t in sorted(row_indices,key=lambda t:t[1])]
    sorted_matrix = matrix[np.ix_(row_indices)]
    # Prepare the doc_labels for the x-axis of any visualisation
    doc_labels = [clean_name(t[1]) + ' (' + str(t[2]) + ')' for t in doc_data]

    # Return the sorted matrix and the ordered indices of the summarisation topics
    return sorted_matrix,row_indices,doc_labels

def visualise_summarisation(summarisation_choices,summarisation_dict,model_dict,config):

    def get_dot_sizes(matrix_row):
        base_size = 10
        sizes = []
        for v in matrix_row:
            if v > 0:
                sizes.append(base_size*v)
        return sizes  
    
    summarisation_matrix,row_indices,doc_labels =\
        get_summarisation_matrix(summarisation_dict,model_dict,config)
    
    f = plt.figure(figsize=(16,12))
    topic_counter = 0
    topic_labels = list(summarisation_dict.keys())
    for i,row_index in enumerate(row_indices):
        component = summarisation_dict[topic_labels[row_index]]
        if len(component) < summarisation_choices['min_size'] or len(component) > summarisation_choices['max_size']:
            continue

        topic_counter += 1
        row = summarisation_matrix[i]
        sizes = get_dot_sizes(row)
        x = [j for j,x in enumerate(row) if x > 0]
        y = [i]*len(x)
        plt.scatter(x,y,alpha=0.8,linewidth=0.5,s=sizes)
        plt.plot(x,y,alpha=0.8,linewidth=0.5)

    print('Total number of summarisation topics:',summarisation_matrix.shape[0])
    print('Number of selected summarisation topics:',topic_counter)
    
    plt.ylim(-1,summarisation_matrix.shape[0])
    plt.xticks(range(0,len(doc_labels)),doc_labels,rotation=90,fontsize='x-large')
    plt.ylabel('Summaristion topic index in order of first appearance',fontsize='x-large')
    plt.xlabel('Document name and date in date order',fontsize='x-large')
    plt.show()

def get_sorted_component(component,model_dict,config):
    """
    Sort segments in ascending date order of documents
    param component: unsorted component
    param model_dict: data model
    param model_keys: data model keys
    return: sorted component and label for earliest document in component
    """
    name_field = config['document_metadata']['name'][1]
    date_field = config['document_metadata']['date'][1]

    doc_data = []
    for t in component:
        # Convert compoent segment indices into segment IDs
        doc_id = t[0].split('/')[0]
        doc_data.append((t,model_dict[config['model_keys']['documents']][doc_id][name_field].strip(),convert_date(model_dict[config['model_keys']['documents']][doc_id][date_field]),\
                                                                                                model_dict[config['model_keys']['documents']][doc_id][date_field]))
    doc_data = sorted(doc_data,key=lambda t:t[2])

    # The earliest document in the component
    earliest_doc_label = doc_data[0][1] + ' (' + str(doc_data[0][3]) + ')'

    # Recover the date sorted component
    component = [t[0] for t in doc_data]
    return component,earliest_doc_label


def list_summarisations(summarisation_dict,model_dict):    
    """
    List summarisations in HTML.
    Summary indices define the order of first appearance of the summarisation topics
    """

    html = ''
    topic_counter = 0

    for topic_label,component in summarisation_dict.items():
        if len(component) > 100:
            continue
        summary = component[np.argmax([t[1] for t in component])]
        topic_counter += 1

        # Header table
        html += '<table style="border: 1px solid grey; width: 100%;">'
        html += '<tr>'
        html += '<td style="text-align:left;background-color:lightblue;width:62%;">\
             &nbsp;|&nbsp;Topic ID:&nbsp;' + str(topic_label) +\
        '&nbsp;|&nbsp;Segments in topic:&nbsp;' + str(len(component)) + '</td>'
        html += '</tr>'
        html += '</table>'

        html += '<table style="border: 1px solid grey; width: 100%;">'
            
        for segment_tuple in component:
            segment_id = segment_tuple[0]
            segment_text = model_dict['segments_dict'][segment_id]['text']
            if segment_id in summary:
                bg_color = 'pink'
            else:
                bg_color = ''
            html += '<tr>'
            html += '<td style="word-wrap:break-word;text-align:left;background-color:' + bg_color + ';width:62%;">' + segment_text + '</td>'
            html += '<td style="word-wrap:break-word;text-align:left;background-color:' + bg_color + ';width:8%;">' + segment_id + '</td>'

            html += '</tr>'
            
        html += '</table>'
        html += '<p>&nbsp;</p>'
    print('Number of selected topics:',topic_counter)
    print()
    display(HTML(html))
    return topic_counter


def export_summarisation(summarisation_dict,model_dict,file_name=''):
    """
    Export summarisation topics to CSV.
    param summarisation_dict: summarisation dictionary containing summary and component
    param summarisation_choices: UI selections
    param model_dict: data model
    """
    num_topics = len(summarisation_dict)

    csv_row_list = [] 

    header = []
    header.append('Topic ID')
    header.append('Segment is summary')
    header.append('Segment text')
    header.append('Segment ID')
    csv_row_list.append(header)
    
    topic_labels = list(summarisation_dict.keys())
    for topic_label,component in summarisation_dict.items():
        if len(component) > 100:
            continue
        summary = component[np.argmax([t[1] for t in component])]

        for segment_tuple in component:
            segment_id = segment_tuple[0]
            segment_text = model_dict['segments_dict'][segment_id]['text']
            
            is_summary = False
            if segment_id in summary:
                is_summary = True

            csv_row = []
            csv_row.append(topic_label)
            csv_row.append(is_summary)
            csv_row.append(segment_text)
            csv_row.append(segment_id)
            csv_row_list.append(csv_row)
        csv_row = []
        csv_row.append('')
        csv_row_list.append(csv_row)
            
    if len(file_name) > 0:
        with open('./outputs/' + file_name + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_row_list)
        f.close()

"""
Code below generates interfaces
"""

def init_summarisation_choices(int_nat=False):
    choice_dict = {}
    choice_dict['min_similarity'] = 0.7
    choice_dict['max_similarity'] = 0.8
    choice_dict['min_size'] = 2
    choice_dict['max_size'] = 2
    choice_dict['export'] = False
    choice_dict['verbose'] = False
    choice_dict['export_name'] = ''
    return choice_dict

def generate_summarisation_interface(summarisation_choices,defaults=[0.7,0.8]):
    """
    Interface to set the cluster threshold
    """
    def apply_func(change):
        summarisation_choices['min_similarity'] = threshold_slider.value[0]
        summarisation_choices['max_similarity'] = threshold_slider.value[1]

    threshold_slider = widgets.FloatRangeSlider(
        value=defaults,
        min=0.6,
        max=1.0,
        step=0.01,
        description='Search:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=Layout(width='800px')
    )
    
    summarisation_apply = widgets.Button(
        description='Apply Choices',
        disabled=False,
        button_style='',
        tooltip='Click to apply choices'
    )

    summarisation_apply.on_click(apply_func)
    display(threshold_slider)
    display(summarisation_apply)

    out = widgets.Output()
    display(out)

def generate_display_interface(summarisation_choices,size_options):

    def apply_func(change):
        if len(size_choices.value) == 0:
            min_value = size_options[0]
            max_value = size_options[-1]
        else:
            min_value = min(size_choices.value)
            max_value = max(size_choices.value)
        summarisation_choices['min_size'] = min_value
        summarisation_choices['max_size'] = max_value
        summarisation_choices['verbose'] = verbose_checkbox.value
        export_name = '' 
        if export_checkbox.value == True:
            export_name = export_filename.value
            if len(export_name.strip()) == 0 or export_name == None:
                alert('Please enter an export file name')
                return
            summarisation_choices['export'] = export_checkbox.value 
            summarisation_choices['export_name'] = export_name 


    size_choices = widgets.SelectMultiple(
        options=size_options,
        value=[],
        rows=10,
        description='Size range:',
        disabled=False
    )

    verbose_checkbox = widgets.Checkbox(
        value=False,
        description='Verbose:',
        disabled=False,
        indent=True    
    )

    export_checkbox = widgets.Checkbox(
        value=False,
        description='Export:',
        disabled=False,
        indent=True    
    )

    export_filename = widgets.Text(
        layout={'width': 'initial'},
        value='',
        placeholder='Enter file name',
        description='Export name:',
        disabled=False,
        continuous_update=False
    )

    display_apply = widgets.Button(
        description='Apply Choices',
        disabled=False,
        button_style='',
        tooltip='Click to apply choices'
    )

    display_apply.on_click(apply_func)
    display(size_choices)
    display(verbose_checkbox)
    display(export_checkbox)
    display(export_filename)
    display(display_apply)

