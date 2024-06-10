from django.shortcuts import render
import os
# our home page view
def home(request):    
    return render(request, 'index.html')


# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    import pickle
    # Get the absolute path of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the model file
    model_path = os.path.join(base_dir, 'titanic_survival_ml_model.sav')
    scaler_path = os.path.join(base_dir, 'scaler.sav')

    # Load the model
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    # Load the scaler
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    #model = pickle.load(open("titanic_survival_ml_model.sav", "rb"))
    #scaled = pickle.load(open("scaler.sav", "rb"))
    prediction = model.predict(scaler.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))
    
    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"
        

# our result page view
def result(request):
    pclass = int(request.GET['pclass'])
    sex = int(request.GET['sex'])
    age = int(request.GET['age'])
    sibsp = int(request.GET['sibsp'])
    parch = int(request.GET['parch'])
    fare = int(request.GET['fare'])
    embC = int(request.GET['embC'])
    embQ = int(request.GET['embQ'])
    embS = int(request.GET['embS'])

    result = getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS)

    return render(request, 'result.html', {'result':result})