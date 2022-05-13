import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pickle
#import tensorflow as tf
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import dash_bootstrap_components as dbc
from PIL import Image

################################################################################
# APP INITIALIZATION
################################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO]) # Superhero VAPOR

# this is needed by gunicorn command in procfile
server = app.server


################################################################################
# image
################################################################################
#pyLogo = Image.open("logo.png")
# Create figure



# ################################################################################
# # LAYOUT
# ################################################################################
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
            style={"width": "80%", "height": 100, "background-color":'#ADD8E6',"margin":"0 auto", "display":"block"},
        ),
    html.Br(),
    html.Button(
        "Detect Toxicity", 
        id="submit-button-state", 
        n_clicks=0,
        style={"background-color":"#FF5A36","margin":"0 auto", "display":"block", 'padding': 10}
        ),
    html.Div(id='my-output',style={'text-align': 'center', 'padding': 100}),
    html.Div(id="textarea-state-example-output", style={"whiteSpace": "pre-line"}),
    # html.Img(src=pyLogo, width="500", height="120",
    #             style={
    #             "position": "absolute", "bottom":"3px", "right":"3px", "padding-right": "20px"}) 
])

#"position": "relative", "float": "right", "padding-right": "20px"
################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input("submit-button-state", "n_clicks"),
    State(component_id='my-input', component_property='value'),
)


def update_output(n_clicks, input_value):
    if n_clicks > 0:

        # with open('./models/tokenizer.pickle', 'rb') as handle:
        #     tokenizer = pickle.load(handle)
        # with open('./models/model.pickle', 'rb') as handle:
        #     model = pickle.load(handle)
        
        #sequences_y = tokenizer.texts_to_sequences([input_value])
        #data_y = pad_sequences(sequences_y, padding = 'post', maxlen = 200)
        #y_hat = model.predict(data_y)
        # if y_hat[0][0]>=0.5:
        #     output_value = "The text is toxic!"
        # else:
        #     output_value = "No toxicity detected."
        # print(y_hat[0][0])
        return 'hello' #output_value

# Add the server clause:
if __name__ == "__main__":
    app.run_server()