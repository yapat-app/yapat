# notes
'''
This file is for creating a navigation bar that will sit at the top of your application.
Much of this page is pulled directly from the Dash Bootstrap Components documentation linked below:
https://dash-bootstrap-components.opensource.faculty.ai/docs/components/navbar/
'''

import dash_bootstrap_components as dbc
# package imports
from dash import callback, Output, Input, State

# local imports
# from utils.images import logo_encoded
# from components.login import login_info

# component
# navbar = dbc.Navbar(
#     dbc.Container(
#         [
#             dbc.NavbarToggler(id='navbar-toggler', n_clicks=0),
#             dbc.Collapse(
#                 dbc.NavbarSimple(
#                     children=[
#                         dbc.NavItem(dbc.NavLink('Project', id='nav_datasource', href='/project')),
#                         dbc.NavItem(dbc.NavLink('Settings', id='nav_settings', href='/settings', disabled=True)),
#                         dbc.NavItem(dbc.NavLink('Annotation', id='nav_annotation', href='/annotate', disabled=False)),
#                         dbc.NavItem(dbc.NavLink('Reset', id='nav_exit', href='/exit', disabled=True))
#                     ],
#                     brand='YAPAT',
#                     brand_href='/',
#                     color='dark',
#                     dark=True,
#                     id='navbar'
#                 ),
#                 id='navbar-collapse',
#                 navbar=True
#             ),
#         ]
#     ),
#     color='dark',
#     dark=True,
# )


navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink('Project', id='nav_datasource', href='/project')),
                dbc.NavItem(dbc.NavLink('Settings', id='nav_settings', href='/settings', disabled=True)),
                dbc.NavItem(dbc.NavLink('Annotation', id='nav_annotation', href='/annotate', disabled=False)),
                dbc.NavItem(dbc.NavLink('Reset', id='nav_exit', href='/exit', disabled=True))
            ],
            brand='YAPAT',
            brand_href='/',
            color='dark',
            dark=True,
            id='navbar'
        )

# add callback for toggling the collapse on small screens
# @callback(
#     Output('navbar-collapse', 'is_open'),
#     Input('navbar-toggler', 'n_clicks'),
#     State('navbar-collapse', 'is_open'),
# )
# def toggle_navbar_collapse(n, is_open):
#     if n:
#         return not is_open
#     return is_open
