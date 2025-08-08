from supabase_client import supabase

def log_to_supabase(filename, prediction, feedback_dict, exercise_type):
    # score'u güvenle çıkar (rule_based içinde)
    score = (
        feedback_dict.get("score") or
        (feedback_dict.get("rule_based") or {}).get("score") or
        (feedback_dict.get("llm") or {}).get("score")  # ileride eklersen
    )

    try:
        supabase.table("predictions").insert({
            "filename": filename,
            "exercise_type": exercise_type,
            "prediction": prediction,
            "score": score,                 
            "feedback": feedback_dict,      
            "keypoint": None
        }).execute()
        print("Supabase log success.")
    except Exception as e:
        print(f"Supabase log failed: {e}")
