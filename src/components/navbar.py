import dash_bootstrap_components as dbc
from dash import callback, Input, Output

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('Project', id='nav_datasource', href='/project')),
        # dbc.NavItem(dbc.NavLink('Settings', id='nav_settings', href='/settings', disabled=True)),
        dbc.NavItem(dbc.NavLink('Annotation', id='nav_annotation', href='/annotate', disabled=True)),
        # dbc.NavItem(dbc.NavLink('Reset', id='nav_exit', href='/exit', disabled=True))
    ],
    brand='YAPAT',
    brand_href='/',
    color='dark',
    dark=True,
    id='navbar'
)


@callback(
    Output('navbar', 'brand'),
    Output('nav_annotation', 'disabled'),
    Input('project-content', 'data')
)
def update_navbar(data):
    brand = 'YAPAT'
    disabled = True
    if data.get('project_name'):
        brand = f"YAPAT | {data.get('project_name')}"
        disabled = False

    return brand, disabled

# TODO Add users