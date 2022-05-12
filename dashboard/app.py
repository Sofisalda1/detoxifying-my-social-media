import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import dash_bootstrap_components as dbc


################################################################################
# APP INITIALIZATION
################################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR]) # Superhero VAPOR

# this is needed by gunicorn command in procfile
server = app.server


# ################################################################################
# # PLOTS
# ################################################################################

################################################################################
# LAYOUT
################################################################################
app.layout = html.Div([
        html.H2(
            id="title",
            children="Detoxify My Social Media",
            style={'text-align': 'center'}
        ),
        html.Div(
        "Add your text to see if it's toxic:", 
        style={'text-align': 'center'}),
        dcc.Textarea(
            id="my-input",
            value="",
            style={"width": "100%", "height": 100, "background-color":'cyan'},
        ),
    html.Br(),
    html.Button(
        "Detect Toxicity", 
        id="submit-button-state", 
        n_clicks=0,
        style={"background-color":"#FF5A36", "justify":"right"}
        ),
    html.Div(id='my-output',style={'text-align': 'center'}),
    html.Div(id="textarea-state-example-output", style={"whiteSpace": "pre-line"}),
])


app = dash.Dash(__name__)


################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input("submit-button-state", "n_clicks"),
    State(component_id='my-input', component_property='value'),
)



def update_output_div(n_clicks, input_value):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('MODEL.pickle', 'rb') as handle:
        model = pickle.load(handle)
    if n_clicks > 0:
        sequences_y = tokenizer.texts_to_sequences([input_value])
        data_y = pad_sequences(sequences_y, padding = 'post')
        y_hat = model.predict(data_y)
        if y_hat[0][0]>=0.8:
            output_value = "The text is toxic!"
        else:
            output_value = "No toxicity detected."
        return output_value


# Add the server clause:
if __name__ == "__main__":
    app.run_server()