import argparse

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

import os
import webbrowser

# Initialize the Dash app
app = dash.Dash(__name__)

# loads and parses match history data from scraped csv
# def load_match_history(filename=args.file):
def load_match_history(filename="match_history.csv"):
    data = pd.read_csv(filename)

    data["Event Date"] = pd.to_datetime(data["Event Date"].map(fix_fab_date))
    
    # THIS IS EXTREMELY INEFFICIENT - MUST TIDY UP
    data["Opponent"] = data["Opponent"].apply(lambda x: get_most_recent_name_by_gem_id(data, str(x)))

    data["Result"] = data["Result"].apply(lambda x: False if x == "Loss" else True)

    data = data[["Opponent", "Result", "Rated", "Round", "Event Date"]]
    return data

# fix the shitty fab dates so they can be parsed
def fix_fab_date(date):
    date = date.replace("Jan.", "January")
    date = date.replace("Feb.", "February")
    date = date.replace("Aug.", "August")
    date = date.replace("Sept.", "September")
    date = date.replace("Oct.", "October")
    date = date.replace("Nov.", "November")
    date = date.replace("Dec.", "December")

    date = "".join(date.split(",")[:-1])

    return date

# fetches the most recently used name from the match history for a gem id
def get_most_recent_name_by_gem_id(data, id): 
    data_for_name = data[(data["Opponent"] == id)]
    row = data_for_name[data_for_name["Event Date"] == data_for_name["Event Date"].max()].iloc[0]
    return row["Opponent"]

#### Script Here ####

"""
What data do we want?
 - opponent data (win rates, matches played, visualisations)
 - round data
 - time-based data? (w/r per month, year)
 
What filters do we want?
 - date filter (all time, past year, past 6mths, past month)
 - # games for opponents / visualisations (exclude 1of games)
 - rated vs unrated vs all
"""

def init_layout(app, filename):
    # Define the layout of the app
    app.layout = html.Div([
        dcc.Store(id='match_history', data=load_match_history().to_dict("records")),
        dcc.Store(id="match_history_filtered"),
        dcc.Store(id="opponent_history"),

        html.Div(className="header", children=[
            html.H3('FaB History Analysis', className="title"),
            html.H4('Interactive visualization of matchup history.', className="subtitle"),
            html.Div([
                html.Div("Rated Matches?"),
                dcc.RadioItems(
                    id='rated_filter',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Rated', 'value': 'True'},
                        {'label': 'Unrated', 'value': 'False'}
                    ],
                    value='all',  # Default value
                ),
            ], className="filters"),
        ]),

        html.Div(className="content", children=[

            ### General Stats
            html.Div([
                html.H2("General Stats"),
                html.Hr(),
                html.Div(id='stats_output'),  # Div to display updated stats
            ], className="statbox"),
            
            ### Opponent Stats
            html.Div([
                html.H2("Opponent Stats"),
                html.H3("Opponent Filters"),
                html.Div([
                    html.Div("Min. Games"),
                    dcc.Input(id='min_games_filter', type='number', min=1, step=1, value='1'),
                ]),

                html.H3("Search for an opponent:"),
                html.Div([
                    html.Label("Search for an opponent: "),
                    dcc.Input(id='opponent_name_input', type='text', placeholder='Enter opponent name'),
                    html.Br(),html.Br(),
                    html.Div(id='opponent_name_output')
                ]),

                
                html.H3("Win Rates for Opponents"),
                dcc.Dropdown(
                    id='sort_by_dropdown',
                    options=[
                        {'label': 'Name - Ascending', 'value': 'Name_asc'},
                        {'label': 'Name - Descending', 'value': 'Name_desc'},
                        {'label': 'Win Rate - Ascending', 'value': 'Win Rate_asc'},
                        {'label': 'Win Rate - Descending', 'value': 'Win Rate_desc'}
                    ],
                    value='Name_asc'
                ),html.Br(),
                dcc.Graph(id='fig_win_rates'),

                html.H3("Most Played Opponents"),
                dcc.Graph(id='fig_top_opponents'),
            ], className="statbox"),
            
            ### Round Stats
            html.Div([
                html.H2("Round Stats"),
                dcc.Graph(id='fig_round_win_rate')
            ], className="statbox"),     
        ])
    ])



# This callback just updates the global filtered data based on the existing
# filters provided by the users
@app.callback(
        Output('match_history_filtered', 'data'),
        [Input('match_history', 'data'), Input('rated_filter', 'value')]
)
def update_match_history_filtered(match_history, rated_filter):
    match_history = pd.DataFrame(match_history)

    if rated_filter == 'all':
        filtered_data = match_history
    elif rated_filter == 'True':
        filtered_data = match_history[match_history['Rated'] == True]
    else:
        filtered_data = match_history[match_history['Rated'] == False]

    return filtered_data.to_dict("records")

@app.callback(
        Output('opponent_history', 'data'),
        [Input('match_history_filtered', 'data'), Input('min_games_filter', 'value')]
)
def update_opponent_history(match_history, min_games):
    match_history = pd.DataFrame(match_history)
    
    min_games = int(min_games) if min_games else 1
    
    # set up opponent data
    opponent_stats = match_history.groupby('Opponent').agg(
        Match_Count=('Opponent', 'size'),                
        Win_Count=('Result', lambda x: x.sum()),       # Count the wins
        Loss_Count=('Result', lambda x: (1-x).sum()),  # Count the losses
    ).reset_index()
    opponent_stats['Win_Rate'] = opponent_stats['Win_Count'] / opponent_stats['Match_Count']

    # apply min games filter
    opponent_stats = opponent_stats[opponent_stats['Match_Count'] >= min_games]

    return opponent_stats.to_dict("records")

@app.callback(
    Output('stats_output', 'children'),
    [Input('match_history_filtered', 'data')]
)
def update_general_stats(match_history):
    match_history = pd.DataFrame(match_history)

    total_matches = len(match_history)
    total_winrate = match_history['Result'].mean()
    number_opponents = len(match_history.groupby('Opponent'))

    return html.Div([
        html.P(f'Total Matches: {total_matches}'),
        html.P(f'Total Winrate: {total_winrate:.2%}'),
        html.P(f'Number of Different Opponents: {number_opponents}')
    ])


@app.callback(
    Output('opponent_name_output', 'children'),
    [Input('opponent_history', 'data'), Input('opponent_name_input', 'value')]
)
def update_opponent_search(opponent_history, query):
    opponent_history = pd.DataFrame(opponent_history)

    if query is None or query == "":
        return 'Enter an opponent name or GEM ID and click submit.'

    filtered_data = opponent_history[opponent_history["Opponent"].str.contains(query, case=False)]

    if filtered_data.empty:
        return 'No opponents matching this query.'

    # Formatting the output
    return html.Ul([
        html.Li(f"{row['Opponent']}: {row['Match_Count']} matches, {row['Win_Rate']:.2%} win rate") 
        for _, row in filtered_data.iterrows()
    ])

# THIS SHOULD JUST BE A TABLE????????????
@app.callback(
    Output('fig_win_rates', 'figure'),
    [Input('opponent_history', 'data'), Input('sort_by_dropdown', 'value')]
)
def update_graph(opponent_history, sort_by_value):
    opponent_history = pd.DataFrame(opponent_history)

    # Convert 'Win_Rate' to percentage
    opponent_history['Win_Rate'] *= 100

    # Determine sorting
    sort_by, order = sort_by_value.split('_')
    ascending = order == 'asc'

    if sort_by == 'Name':
        opponent_history.sort_values(by='Opponent', ascending=ascending, inplace=True)
    elif sort_by == 'Win Rate':
        opponent_history.sort_values(by='Win_Rate', ascending=ascending, inplace=True)

    # Create the figure
    figure = px.bar(
        opponent_history,
        x='Opponent',
        y='Win_Rate',
        title='Win Rate Against Each Opponent'
    )

    # Update hover template to show percentage and match count
    figure.update_traces(
        hovertemplate='Opponent: %{x}<br>Win Rate: %{y:.2f}%<br>Match Count: %{customdata}'
    )
    figure.update_layout(yaxis_title="Win Rate (%)")

    # Add customdata for hover info
    figure.update_traces(customdata=opponent_history['Match_Count'])

    return figure



@app.callback(
        Output('fig_top_opponents', 'figure'),
        [Input('opponent_history', 'data')]
)
def create_figure_top_opponents(opponent_history, n=5):
    opponent_history = pd.DataFrame(opponent_history)

    top_opponents = opponent_history.nlargest(n, 'Match_Count')

    # Assuming top_5_opponents is already sorted by 'Match_Count' descending
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_opponents['Opponent'],
        y=top_opponents['Win_Count'],
        name='Wins',
        marker_color='#6495ED'
    ))
    fig.add_trace(go.Bar(
        x=top_opponents['Opponent'],
        y=top_opponents['Loss_Count'],
        name='Losses',
        marker_color='#FF7F24'
    ))

    # Modify the layout to stack the bars
    fig.update_layout(
        barmode='stack',
        title='Top {} Opponents by Match Count'.format(n),
        xaxis_title='Opponent',
        yaxis_title='Count'
    )

    return fig



@app.callback(
        Output('fig_round_win_rate', 'figure'),
        [Input('match_history_filtered', 'data')]
)
def create_figure_round_win_rate(match_history):
    match_history = pd.DataFrame(match_history)

    # Group data by round and calculate win rate
    round_win_rate = match_history.groupby('Round')['Result'].mean().reset_index()
    round_win_rate.rename(columns={'Result': 'Win Rate'}, inplace=True)

    round_win_rate['Win Rate'] = round_win_rate['Win Rate'] * 100
    round_win_rate_figure = px.bar(round_win_rate, x='Round', y='Win Rate', title='Win Rate per Round')

    # Update hover template to show percentage
    round_win_rate_figure.update_traces(hovertemplate='Round %{x}<br>Win Rate=%{y:.2f}%')
    round_win_rate_figure.update_layout(yaxis_title="Win Rate (%)")

    return round_win_rate_figure



# Run the app
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default="match_history.csv",
                        help="the path to your FaB history as scraped by GreaseMonkey/TamperMonkey: please see README for instructions on how to obtain this")
    args = parser.parse_args()

    # Initialise the Dash layout using the provided data file
    init_layout(app, args.file)

    # The reloader has not yet run - open the browser
    # https://stackoverflow.com/questions/54235347/open-browser-automatically-when-python-code-is-executed
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8050/')

    # Start the Flask App
    app.run_server(debug=False, host='0.0.0.0', port=8050)
