from doctest import debug
import gradio as gr
import pickle

imported_model = pickle.load(open('Model1.pkl', 'rb'))

def pred_servival(gutFeel, sex, PassengerId, Pclass, Age, SibSp, Parch, Fare):
    print(f" Input Data\n:gutFeel: {gutFeel}, Sex: {sex}, PassengerId: {PassengerId}, Pclass: {Pclass}, Age: {Age}, SibSp: {SibSp}, Parch: {Parch}, Fare: {Fare}")
    print(f"Prediction: {imported_model.predict([[gutFeel, sex,PassengerId, Pclass, Age, SibSp, Parch, Fare]])[0]}")
    if (imported_model.predict([[gutFeel, sex,PassengerId, Pclass, Age, SibSp, Parch, Fare]])[0] == 1):
        return "Passenger Survived"
    else:
        return "Passenger Not Survived"

ui = gr.Interface(fn=pred_servival, inputs=["number","number","number","number","number","number","number","number"], outputs="text")


if __name__ == "__main__":
    ui.launch(debug=True)