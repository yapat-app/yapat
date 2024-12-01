import dash_bootstrap_components as dbc
from dash import callback, Input, Output

from src.components.login import login_info

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('Home', id='nav_home', href='/')),
        dbc.NavItem(dbc.NavLink('Explore', id='nav_explore', href='/explore', disabled=False)),
        dbc.NavItem(dbc.NavLink('Compare', id='nav_compare', href='/compare', disabled=True)),
        dbc.NavItem(dbc.NavLink('Annotate', id='nav_annotate', href='/annotate', disabled=True)),
        login_info
    ],
    brand='YAPAT',
    brand_href='/',
    color='dark',
    dark=True,
    id='navbar'
)


@callback(
    Output('navbar', 'brand'),
    Output('nav_annotate', 'disabled'),
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