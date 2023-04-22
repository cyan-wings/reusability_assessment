from classifier import CLF
from flask import Flask, render_template, request, Markup
import dill as pickle
import os
import shutil
import pandas as pd
import shutil

def remove_columns(df):
   #Uses for loop to skip any error in dropping columns that don't exist (Some files may not have columns with a subset of these columns)
   columns_to_remove = ['ID', 'Name', 'LongName', 'Parent', 'Component', 'Path', 'Line', 'Column', 'EndLine', 'EndColumn', 'WarningBlocker', 'WarningCritical', 'WarningInfo', 'WarningMajor', 'WarningMinor', 'Anti Pattern', 'Design Pattern']
    
   for col in columns_to_remove:
      try:
         df = df.drop(columns=[col], axis=1)
      except:
         pass
    
   return df

rankingData = pd.read_csv("forWebsite.csv", sep=",", encoding='utf-8')

classifier_dict = {
   "Default": "Default",
   "RF": "Random Forest",
   "MLP": "Multi-Layer Perceptron",
   "SVM": "Support Vector Machine (RBF)",
   "KNN": "K-Nearest Neighbors",
   "SGD": "Logistic Regression",
   "DT": "Decision Trees",
   "R": "Ridge Regression",
   "XG": "Extreme Gradient Boosting",
   "GPC": "Gaussian Process",
   "ADA": "Adaptive Boosting",
   "GB": "Gradient Boosting"
}

app = Flask(__name__)
@app.route('/')
def indexPage(): 
   return githubSearchPage()

@app.route('/github_search', methods = ['GET'])
def githubSearchPage():
   return render_template('index.html', currentModel = "Default", modelData=classifier_dict)

@app.route('/ranking', methods = ['GET', 'POST'])
def rankingPage():
   currentTag = 'All'
   select = request.form.get('tag')
   if select != None:
      currentTag = select

   if select != 'All' and select != None:
      sortedRanking = rankingData[rankingData[select] == 1].sort_values(by=['predicted'], ascending=False).reset_index()
      
   else:
      sortedRanking = rankingData.sort_values(by=['predicted'], ascending=False).reset_index()
   
   if select == None:
      select = 'All'

   tagsForSelection = []
   for t in rankingData.iloc[:, 4:-2].columns:
      tagsForSelection.append(t)
   return render_template('ranking.html', currentTag = currentTag, data = sorted(tagsForSelection), tag = select, rankingTable = sortedRanking.head(10))

@app.route('/visualisations')
def visualisationsPage():
   return render_template('visualisations.html')

@app.route('/about')
def aboutPage():
   return render_template('about.html')

@app.route('/predict', methods = ['POST'])
def predict():
   currentModel = "Default"

   gitLink = request.form["githublink"]
   selectedModel = request.form["modelSelect"]
   if selectedModel == "Default":
      selectedModel = "R"
   else:
      currentModel = classifier_dict[selectedModel]
   model = pickle.load(open('Saved_Models/{}.pkl'.format(selectedModel),'rb'))

   token = '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'
   flag = os.system("git clone https://{}:x-oauth-basic@github.com/{}".format(token, gitLink))

   if flag==32768:
      shutil.rmtree(gitLink.split('/')[1])
      return render_template('index.html', modelData=classifier_dict, pre_text='Input a valid GitHub Link')

   flag = os.system("SourceMeter-9.1.1/Java/SourceMeterJava " + "-projectName=test " + "-resultsDir=r -projectBaseDir={} ".format(gitLink.split('/')[1]) + "-runAndroidHunter=false -runMetricHunter=false -runVulnerabilityHunter=false -runFaultHunter=false -runRTEHunter=false -runDCF=false -runMET=true -currentDate=none -runPMD=false -runFB=false")
   #except IndexError:
      #return render_template('index.html', modelData=classifier_dict, pre_text='Input a valid GitHub Link')
        
   if flag == 0:

      os.system('mv r/test/java/none/test-Class.csv r/test')
      os.system('mv r/test/java/none/test-File.csv r/test')
      os.system('mv r/test/java/none/test-Method.csv r/test')
      shutil.rmtree('r/test/java', ignore_errors=True)
        
   else:
      print('error')
      #error_array.append(i)

   try:
      granularities = ['Class', 'File', 'Method']
      test_sample_metric_names = []
      test_sample_array = []
      for g in granularities:
         df = pd.read_csv('r/test/test-{}.csv'.format(g), encoding='utf-8')
         df = remove_columns(df)
         test_sample_array.append(len(df.index))   #For No_C, No_F and No_M
         test_sample_metric_names.append('No_{}'.format(g[0]))
         for col in df.columns:
            test_sample_array.append(df[col].min())
            test_sample_array.append(df[col].median())
            test_sample_array.append(df[col].max())
            test_sample_array.append(df[col].sum())
            test_sample_array.append(df[col].std())

            test_sample_metric_names.append('{}min_{}'.format(g[0], col))
            test_sample_metric_names.append('{}med_{}'.format(g[0], col))
            test_sample_metric_names.append('{}max_{}'.format(g[0], col))
            test_sample_metric_names.append('{}sum_{}'.format(g[0], col))
            test_sample_metric_names.append('{}std_{}'.format(g[0], col))
   except FileNotFoundError:
      print('File not found')
   
   sample_array = pd.Series(data=test_sample_array, index=test_sample_metric_names)
   #print(sample_array['No_F'])
   sample_array = sample_array.drop(['Cmin_NOC', 'Cmed_NOC', 'Cmin_NOD', 'Cmed_NOD', 'Cmin_NLPA', 'Cmin_NLS', 'Cmin_NPA', 'Cmin_NS', 'Cmin_TNLPA', 'Cmin_TNLS', 'Cmin_TNPA', 'Cmin_TNS', 'Fmin_CLOC', 'Fmed_CLOC', 'Fmax_CLOC', 'Fsum_CLOC', 'Fstd_CLOC', 'Mmin_NL', 'Mmin_NLE', 'Mmin_NII', 'Mmin_NOI', 'Mmin_CD', 'Mmin_CLOC', 'Mmin_DLOC', 'Mmin_TCD', 'Mmin_TCLOC', 'Cmin_CLOC', 'Cmin_DLOC', 'Cmin_TCLOC', 'Cmed_NOP', 'Cstd_NOP', 'Cmed_NLG', 'Cmed_TNLG', 'Mmin_HPL', 'Mmin_HPV', 'Mmin_HTRP', 'Mmin_MI'])
   #print(model.clf.predict([sample_array]))

   if sample_array.No_C == 0:
      shutil.rmtree(gitLink.split('/')[1])
      return render_template('index.html', modelData=classifier_dict, pre_text='Input a valid Java Project')

   if model.clf.predict([sample_array])[0] == 1:
      prediction = Markup('<mark style="background-color:initial;color:#090" class="has-inline-color">HIGH</mark>')
   else:
      prediction = Markup('<mark style="background-color:initial;color:#FF0000" class="has-inline-color">LOW</mark>')

   status_array = []
   importance_features = [('Fsum_PUA', 40, 'H'), ('No_F', 250, 'L'), ('Cmax_NII', 80, 'L'), ('Csum_NL', 450, 'L'), ('Cstd_CBO', 2, 7),  ('Cmax_LCOM5', 10, 'H'), ('Cmax_CBO', 20, 80)]
   actual_values = []
   for f in importance_features:
      actual_values.append(sample_array[f[0]])
      if f[2] == 'L':
         if sample_array[f[0]] < f[1]:
            status_array.append('bx bxs-check-circle icon-green')
         else:
            status_array.append('bx bxs-chevron-down icon-red')
      elif f[2] == 'H':
         if sample_array[f[0]] > f[1]:
            status_array.append('bx bxs-check-circle icon-green')
         else:
            status_array.append('bx bxs-chevron-up icon-red')
      else:
         if sample_array[f[0]] >= f[1] and sample_array[f[0]] <= f[2]:
            status_array.append('bx bxs-check-circle icon-green')
         elif sample_array[f[0]] < f[1]:
            status_array.append('bx bxs-chevron-up icon-red')
         elif sample_array[f[0]] > f[2]:
            status_array.append('bx bxs-chevron-down icon-red')

   table_features = Markup('''\
          <div class="card">
            <div class="card-body">
              <h5 class="card-title">Recommended Metric Values</h5>

              <!-- Dark Table -->
              <table class="table table-dark">
                <thead>
                  <tr>
                    <th scope="col">#</th>
                    <th scope="col">Metric</th>
                    <th scope="col">Recommended</th>
                    <th scope="col">Value</th>
                    <th scope="col">Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">1</th>
                    <td>PUA - File (sum)</td>
                    <td>> 40</td>
                    <td>{actual[0]}</td>
                    <td><i class="{status[0]}"/></td>
                  </tr>
                  <tr>
                    <th scope="row">2</th>
                    <td>Number of files</td>
                    <td>< 250</td>
                    <td>{actual[1]}</td>
                    <td><i class="{status[1]}"/></td>
                  </tr>
                  <tr>
                    <th scope="row">3</th>
                    <td>NII - Class (max)</td>
                    <td>< 80</td>
                    <td>{actual[2]}</td>
                    <td><i class="{status[2]}"/></td>
                  </tr>
                  <tr>
                    <th scope="row">4</th>
                    <td>NL - Class (sum)</td>
                    <td>< 450</td>
                    <td>{actual[3]}</td>
                    <td><i class="{status[3]}"/></td>
                  </tr>
                  <tr>
                    <th scope="row">5</th>
                    <td>CBO - Class (std)</td>
                    <td>2 - 7</td>
                    <td>{actual[4]}</td>
                    <td><i class="{status[4]}"/></td>
                  </tr>
                  <tr>
                    <th scope="row">6</th>
                    <td>LCOM5 - Class (max)</td>
                    <td>> 10</td>
                    <td>{actual[5]}</td>
                    <td><i class="{status[5]}"/></td>
                  </tr>
                  <tr>
                    <th scope="row">7</th>
                    <td>Cmax_CBO</td>
                    <td>20 - 80</td>
                    <td>{actual[6]}</td>
                    <td><i class="{status[6]}"/></td>
                  </tr>
                </tbody>
              </table>
            </div>
         </div>
   '''.format(actual=actual_values, status=status_array))

   shutil.rmtree(gitLink.split('/')[1])

   return render_template('index.html', currentModel = currentModel, modelData=classifier_dict, pre_text='Reusability is ', prediction_text = prediction, table_display=table_features)

@app.route('/ranking_result', methods = ['POST'])

def rankingResult():
   pass


if __name__ == '__main__':
   
   app.run(host="0.0.0.0", port=5000)