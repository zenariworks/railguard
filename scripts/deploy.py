import roboflow

# Deploy model to Roboflow
rf = roboflow.Roboflow(api_key="YOUR_API_KEY")
rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT").version(1).deploy(model_path="models/custom_model/best.pt")
