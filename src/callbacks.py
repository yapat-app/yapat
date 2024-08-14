import datetime
import os

import dash
import numpy as np
from dash import callback, Input, Output, State
from flask import session

from src import utils, sampling, models

# from src import app_utils
# from src import sampling
# import utils
# # from class_data import data
# from custom_logger import logger
#
# from dash_app import skip_and_submit


def register_callbacks(app):
    @callback(
        Output('checklist_annotation', 'options'),
        Output('checklist_annotation', 'value'),
        Output('input_new_class', 'value'),

        Input('button_add_class', 'n_clicks'),
        Input('text_complete_file_name', 'children'),
        Input('checklist_class', 'value'),
        Input('input_search', 'value'),

        State('checklist_annotation', 'options'),
        State('checklist_annotation', 'value'),
        State('input_new_class', 'value'),
    )
    def update_species_selection(button_add_class, text_file_name, checklist_class, input_search, checklist_options,
                                 checklist_value, input_text_raw):
        # create empty lists if None type
        if checklist_options is None:
            checklist_options = []
        if checklist_value is None:
            checklist_value = []

        # update competence classes
        if checklist_class is not None:
            session['user_data']['competence_classes'] = checklist_class

        # get trigger
        callback_trigger = dash.callback_context.triggered_id

        # evaluate trigger
        if callback_trigger == 'button_add_class':
            input_text = input_text_raw.strip().capitalize()
            if input_text:
                if input_text not in checklist_options:
                    checklist_options.append(input_text)
                if input_text not in checklist_value:
                    checklist_value.append(input_text)
            logger.log_action(datetime.datetime.now(), "", "Add " + input_text, username=None,
                              file_name=session['user_data']['file_path'])
        else:
            annotations = utils.read_annotations()
            checklist_options = utils.get_species_options(annotations, session['user_data']['competence_classes'])
            if callback_trigger == 'text_complete_file_name':
                checklist_value = []

        # exclude options that are not searched for
        if input_search:
            checklist_options = [item for item in checklist_options if input_search.lower() in item.lower()]

        checklist_options.sort()
        checklist_value.sort()

        return checklist_options, checklist_value, ''

    # @callback(
    #     Input('radio_sampling', 'value')
    # )
    # def update_sampling_strategy(sampling_strategy):
    #     if sampling_strategy is not None:
    #         data.update_sampling_strategy(sampling_strategy)

    @callback(
        Input('interval_1s', 'n_intervals'),

        State('radio_sampling', 'value')
    )
    def update_sampling_queue(n_intervals, sampling_strategy):
        # only draw a new sample if the queue is shorter than 4 samples
        if len(session['user_data']['file_index_queue']) < 4:
            annotations = utils.read_annotations()
            sampling_start_time = session['user_data'].get('sampling_start_time')
            new_sampled_tuple, sampling_start_time = sampling.sampling(annotations=annotations,
                                                                       sampling_start_time=sampling_start_time,
                                                                       sampling_strategy=sampling_strategy)
            if new_sampled_tuple is not None:
                session['user_data']['file_index_queue'].append(new_sampled_tuple)

    # TODO bring this back to life
    # @callback(
    #     Output('text_model_suggestions', 'children'),
    #
    #     Input('radio_sampling', 'value'),
    #     Input('checklist_class', 'value'),
    #     Input('text_complete_file_name', 'children'),
    # )
    # def update_model_suggestions(radio_sampling, checklist_class, text_file_name):
    #     if radio_sampling != 'validate' or checklist_class is None:
    #         return ''
    #
    #     # only search in relevant rows
    #     relevant_columns = [col for col in data.detections.columns if
    #                         any(comp_class in col for comp_class in checklist_class)]
    #     # get the detection row used with the relevant columns
    #     detections_row = data.detections.loc[data.file_index, relevant_columns]
    #     # sort the detection rows based on the occurence probability
    #     sorted_detections = detections_row.sort_values(ascending=False)
    #     # specify the number of detections with probability values presented to the user
    #     num_columns = min(3, len(sorted_detections))
    #     top_columns = sorted_detections.head(num_columns)
    #     # create a suggestions string
    #     suggestions = ["Model Suggestions:"] + [f"{col}: {val}" for col, val in top_columns.items()]
    #     # format the string with line breaks
    #     return html.Div(
    #         [html.P(suggestions[0], style={'margin-top': '10px'})] +
    #         [html.Div([line, html.Br()]) for line in suggestions[1:]]
    #     )

    # TODO bring this back to life
    # @callback(
    #     Output('text_species_to_process', 'children'),
    #     Output('row_species_to_process', 'style'),
    #
    #     Input('radio_sampling', 'value')
    # )
    # def update_manual_species_to_process(sampling_strategy):
    #     if sampling_strategy == 'validate':
    #         text = 'Ignore during validation:'
    #         style = {'display': 'block'}
    #     elif sampling_strategy == 'refine':
    #         text = 'Refine only:'
    #         style = {'display': 'block'}
    #     elif sampling_strategy == 'discover':
    #         text = ''
    #         style = {'display': 'none'}
    #     elif sampling_strategy == 'random':
    #         text = ''
    #         style = {'display': 'none'}
    #
    #     return text, style

    # TODO bring this back to life
    # @callback(
    #     Output('dropdown_selected_species', 'value'),
    #     Output('dropdown_selected_species', 'options'),
    #
    #     Input('radio_sampling', 'value'),
    #     Input('checklist_annotation', 'options'),
    #
    #     State('dropdown_selected_species', 'value')
    # )
    # def update_dropdown_selected_species(sampling_strategy, annotation_options, selected_value):
    #     callback_trigger = dash.callback_context.triggered_id
    #
    #     # clear selection if sampling strategy changed
    #     if callback_trigger == 'radio_sampling' or selected_value is None:
    #         selected_value = []
    #
    #     # only select values that are in the dropdown
    #     selected_value = [value for value in selected_value if value in annotation_options]
    #
    #     return selected_value, annotation_options

    # TODO bring this back to life
    # @callback(
    #     Input('dropdown_selected_species', 'value')
    # )
    # def update_sampling_selected_species(dropdown_value):
    #     if dropdown_value is None:
    #         dropdown_value = []
    #     data.update_sampling_selected_species(dropdown_value)

    @callback(
        Output('text_complete_file_name', 'children'),

        Input('button_submit', 'n_clicks'),
        Input('button_skip', 'n_clicks'),

        State('checklist_annotation', 'options'),
        State('checklist_annotation', 'value'),
    )
    def update_screen_file_name(button_submit, button_skip, checklist_options, checklist_value,
                                col_processed='processed', col_skipped='skipped'):

        # get trigger
        callback_trigger = dash.callback_context.triggered_id

        annotations = utils.read_annotations()

        # evaluate trigger
        if callback_trigger == 'button_submit':
            # update annotation file
            file_index = session['user_data']['file_index']
            skip_and_submit.submit_sampled_file(annotations, file_index, checklist_options, checklist_value)
            logger.log_action(datetime.datetime.now(), "", "Submit", username=None,
                              file_name=session['user_data']['file_path'])
        elif callback_trigger == 'button_skip':
            file_index = session['user_data']['file_index']
            skip_and_submit.skip_sampled_file(annotations, file_index, col_skipped=col_skipped)
            logger.log_action(datetime.datetime.now(), "", "Skip", username=None,
                              file_name=session['user_data']['file_path'])

        # TODO This is important: pop from queue
        # temp = skip_and_submit.update_file_properties()
    
        file_index_queue = session['user_data'].get('file_index_queue')
    
        if not file_index_queue:
            logger.info("Queue empty. Drawing random sample")
            sampling_start_time = session['user_data'].get('sampling_start_time')
            sampled_tuple, sampling_start_time = sampling.sampling(annotations, sampling_start_time,
                                                                   sampling_strategy='random',
                                                                   ignore_current_processes=True)
            file_index_queue.append(sampled_tuple)

        new_sampled_tuple = file_index_queue.pop() # Samples are not removed from queue if annotated by another user in the meantime

        # logger.info("Writing to session['user_data']")
        session['user_data']['file_index_queue'] = file_index_queue
        session['user_data']['file_index'] = new_sampled_tuple[0]
        session['user_data']['file_path'] = new_sampled_tuple[1]
        session['user_data']['audio'] = list(new_sampled_tuple[2])
        temp = list(new_sampled_tuple[3])
        session['user_data']['concatenated_audio'] = temp

        return os.path.basename(session['user_data']['file_path'])


    @callback(
        Output('graph_complete_spectrogram', 'figure'),
        Output('graph_complete_spectrogram', 'relayoutData'),

        Input('text_complete_file_name', 'children'),
        Input('switch_audio', 'on'),
        Input('switch_colormap', 'on'),

        State('checklist_annotation', 'options'),
        State('checklist_annotation', 'value'),
        prevent_initial_call=True
    )
    def update_spectr_and_playback(screen_filename, switch_audio, switch_colormap, checklist_options,
                                       checklist_value, col_processed='processed', col_skipped='skipped'):

        # update selected file and get new file name and audio
        if switch_audio:
            selected_audio = 'separation'
        else:
            selected_audio = 'raw'


        # get spectrogram figure
        if switch_colormap:
            colormap = 'black_white'
        else:
            colormap = 'standart'

        if selected_audio == 'raw':
            audio = session['user_data']['audio']
        else:
            audio = session['user_data']['concatenated_audio']
        audio = np.array(audio)

        fig, relayout_data = utils.update_spectrogram(audio, colormap)
        fig.update_layout(
            margin=dict(l=10, r=0, t=10, b=10),  # Adjust margins (left, right, top, bottom)
        )
        return fig, relayout_data

    # @callback(
    #     Output('text_complete_file_name', 'children'),
    #     Output('graph_complete_spectrogram', 'figure'),
    #     Output('graph_complete_spectrogram', 'relayoutData'),
    #
    #     Input('button_submit', 'n_clicks'),
    #     Input('button_skip', 'n_clicks'),
    #     Input('switch_audio', 'on'),
    #     Input('switch_colormap', 'on'),
    # 
    #     State('checklist_annotation', 'options'),
    #     State('checklist_annotation', 'value'),
    # )
    # def update_complete_representation(button_submit, button_skip, switch_audio, switch_colormap, checklist_options,
    #                                    checklist_value, col_processed='processed', col_skipped='skipped'):
    #     # get trigger
    #     callback_trigger = dash.callback_context.triggered_id
    # 
    #     # evaluate trigger
    #     if callback_trigger == 'button_submit':
    #         # update annotation file
    #         annotations = utils.read_annotations()
    #         file_index = session['user_data']['file_index']
    #         skip_and_submit.submit_sampled_file(annotations, file_index, checklist_options, checklist_value)
    #         logger.log_action(datetime.datetime.now(), "", "Submit", username=None,
    #                           file_name=session['user_data']['file_path'])
    #     elif callback_trigger == 'button_skip':
    #         annotations = utils.read_annotations()
    #         file_index = session['user_data']['file_index']
    #         skip_and_submit.skip_sampled_file(annotations, file_index, col_skipped=col_skipped)
    #         logger.log_action(datetime.datetime.now(), "", "Skip", username=None,
    #                           file_name=session['user_data']['file_path'])
    # 
    #     # update selected file and get new file name and audio
    #     if switch_audio:
    #         selected_audio = 'separation'
    #     else:
    #         selected_audio = 'raw'
    #     # file_name, audio = data.get_next_file_properties(selected_audio)
    #     # TODO This is important: pop from queue
    #     # temp = skip_and_submit.update_file_properties()
    # 
    #     file_index_queue = session['user_data'].get('file_index_queue')
    #     # delete all indices from queue that are already annotated
    #     # TODO Vectorize operation
    #     file_index_queue = [tup for tup in file_index_queue if
    #                         annotations.loc[tup[0], col_processed] != 1 and
    #                         annotations.loc[tup[0], col_skipped] != 1]
    # 
    #     # derive next file index from sampling queue
    #     sampling_start_time = session['user_data'].get('sampling_start_time')
    #     if not file_index_queue:
    #         sampled_tuple, sampling_start_time = sampling.sampling(annotations, sampling_start_time,
    #                                                                sampling_strategy='random',
    #                                                                ignore_current_processes=True)
    # 
    #     # get file properties
    #     file_index = file_index_queue[0][0]
    #     file_path = file_index_queue[0][1]
    #     audio = file_index_queue[0][2]
    #     concatenated_audio = file_index_queue[0][3]
    # 
    #     #TODO write to user session
    # 
    #     if selected_audio == 'raw':
    #         audio = session['user_data']['audio']
    #     else:
    #         audio = session['user_data']['concatenated_audio']
    #     file_name = annotations.loc[file_index, 'basename']
    # 
    #     # get spectrogram figure
    #     if switch_colormap:
    #         colormap = 'black_white'
    #     else:
    #         colormap = 'standart'
    #     fig, relayout_data = utils.update_spectrogram(audio, colormap)
    #     fig.update_layout(
    #         margin=dict(l=10, r=0, t=10, b=10),  # Adjust margins (left, right, top, bottom)
    #     )
    #     return file_name, fig, relayout_data

    @callback(
        Output('audio_complete_file', 'src'),

        Input('switch_audio', 'on'),
        Input('text_complete_file_name', 'children'),
        Input('graph_complete_spectrogram', 'relayoutData'),
        prevent_initial_call=True
    )
    def play_audio(switch_audio, file_name, relayout_data):
        # get audio file (np.array shape (nr_samples,)) and sample_rate (int)
        current = datetime.datetime.now()
        logger.log_action(current, "", "Load audio", username=None, file_name=session['user_data']['file_path'])

        if switch_audio:
            selected_audio = 'separation'
        else:
            selected_audio = 'raw'

        if selected_audio == 'raw':
            audio = session['user_data']['audio']
        else:
            audio = session['user_data']['concatenated_audio']
        audio = np.array(audio)

        # # select audio file
        # if switch_audio:
        #     selected_audio = data.concatenated_audio
        # else:
        #     selected_audio = data.audio
        print(f"type audio: {type(audio)}")
        return utils.get_audio_playback(audio, relayout_data)
