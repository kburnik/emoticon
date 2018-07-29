from flask import Flask
from flask import render_template
from common import DataPath
from common import DATA_ROOT_DIR
from dataset import DataSet
from report import Reports


app = Flask(
    __name__,
    template_folder="templates",
    static_folder=DATA_ROOT_DIR,
    static_url_path="/data")

app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route('/')
def index():
  return render_template(
      'index.html',
      datasets=DataPath.names(),
      reports=Reports.all()
      )


@app.route('/dataset')
def datasets():
  return render_template(
      'datasets.html',
      datasets=DataPath.names())


@app.route('/dataset/<name>')
def dataset_single(name):
  dataset = DataSet(
    name=name,
    path=DataPath.get(name),
    config=None)
  return render_template(
      'dataset.html',
      name=name,
      dataset=dataset)


@app.route('/report')
def reports():
  return render_template(
      'reports.html',
      reports=Reports.all())


@app.route('/report/<name>')
def report_single(name):
  return render_template(
      'report.html',
      name=name,
      report=Reports.load_by_name(name))
