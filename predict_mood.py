"""Print an experimental forecast from locally self-reported mood history."""
from mood_engine import next_mood_forecast, read_history

def main():
    result = next_mood_forecast(read_history())
    if not result["available"]:
        print(result["reason"])
        return
    print(f"Last self-reported mood: {result['last_mood']}")
    print(f"Most likely next logged mood pattern: {result['predicted']}")
    print(f"Observed transition share: {result['probability']:.1%}")
    print(result["caveat"])

if __name__ == "__main__":
    main()
