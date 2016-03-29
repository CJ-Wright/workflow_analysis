
from jinja2 import Environment, meta, FileSystemLoader
import os
__author__ = 'christopher'

requirements = ['spring_kwargs', 'imagepath', 'starting_config_comment',
                'experiment', 'pdf_kwargs', 'results', 'potential', 'ensemble',
                'simulation_name']
experiment = {'rstep': 0, 'rmin': 1, 'rmax': 3, 'qbin': 1}
ensemble = {'iterations': 5, 'target_acceptance': .65, 'temperature': 300}
spring_kwargs = {'k': 3}
pdf_kwargs = {'conv': 4}
results = {'start_potential': 50, 'finish_potential': 5}

# Change the default delimiters used by Jinja such that it won't pick up
# brackets attached to LaTeX macros.
report_renderer = Environment(
    block_start_string='%{',
    block_end_string='%}',
    variable_start_string='%{{',
    variable_end_string='%}}',
    loader=FileSystemLoader(os.path.abspath('.'))
)

template = report_renderer.get_template('report_template.tex')

PATH = os.path.dirname(
    os.path.abspath(__file__))  # get the path of current file
template_source = report_renderer.loader.get_source(
    report_renderer, 'report_template.tex')[0]
parsed_content = report_renderer.parse(template_source)
variables = meta.find_undeclared_variables(parsed_content)
print(variables)

# print(template.render(experiment=experiment, ensemble=ensemble,
#                       spring_kwargs=spring_kwargs, pdf_kwargs=pdf_kwargs,
#                       potential='rw', results=results, imagepath='hi'))
# with open('filename', 'w') as output:
#     output.write(template.render(experiment=experiment, ))
