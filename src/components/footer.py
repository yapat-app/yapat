from dash import html

layout = html.Div(
    [
        # Footer section
        html.Footer(
            children=[
                html.P(["© 2024 YAPAT by ", html.A("DFKI", href="https://iml.dfki.de", target="_blank"),
                        ". Licensed under AGPL-3.0."], style={'margin-bottom': '10px'}),
                html.A("Documentation", href="https://yapat.readthedocs.io/", target="_blank",
                       style={'margin-right': '20px'}),
                html.A("GitHub Repository", href="https://github.com/yapat-app/yapat",
                       target="_blank", style={'margin-right': '20px'}),
                html.A("Issue Tracker", href="https://github.com/yapat-app/yapat/issues", target="_blank",
                       style={'margin-right': '20px'}),
                html.A("Contact", href="mailto:thiago.gouvea@dfki.de", target="_blank")
            ],
            style={
                'position': 'fixed', 'bottom': '0', 'width': '100%', 'background-color': '#f1f1f1',
                'text-align': 'center', 'padding': '10px', 'border-top': '1px solid #ccc'
            }
        )
    ]
)
