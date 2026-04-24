import pandas as pd
from src.recommender import build_model, recommend

def main():
    df = pd.read_csv("Movie_data.csv")
    
    similarity = build_model(df)
    
    movie = input("Enter movie name: ")
    
    try:
        results = recommend(movie, df, similarity)
        print("\nRecommended movies:\n")
        for r in results:
            print(r)
    except:
        print("Movie not found. Try exact title.")

if __name__ == "__main__":
    main()

