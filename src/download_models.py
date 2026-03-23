import os
import gdown

def download_models():
    base = os.path.dirname(os.path.abspath(__file__))
    
    best_pt = os.path.join(base, "best.pt")
    keras_model = os.path.join(base, "RealDataModelv2.keras")
    
    if not os.path.exists(best_pt):
        print("Downloading best.pt...")
        gdown.download(
            "https://drive.google.com/uc?id=17tz2sfDAXFhD9KhCZwczkTC_yd4PKTJa",
            best_pt, quiet=False
        )
    
    if not os.path.exists(keras_model):
        print("Downloading RealDataModelv2.keras...")
        gdown.download(
            "https://drive.google.com/uc?id=1Rx8pDoO7o9z_g3aOj4hDPvQMizvpm2Uw",
            keras_model, quiet=False
        )

download_models()