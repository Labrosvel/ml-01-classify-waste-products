# server/server.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import stripe

load_dotenv()  # loads .env during local dev

app = Flask(__name__)
CORS(app)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# The URL of your Streamlit app (use https://... when deployed)
YOUR_DOMAIN = os.getenv("APP_DOMAIN", "http://localhost:8501")


@app.post("/create-checkout-session")
def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "gbp",
                        "unit_amount": 500,  # £5.00 in pence
                        "product_data": {"name": "Waste Classifier Pro"},
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=f"{YOUR_DOMAIN}/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{YOUR_DOMAIN}/cancel",
        )
        # Return the full url to redirect the user directly
        return jsonify({"id": session.id, "url": session.url})
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    # For local testing
    port = int(os.getenv("PORT", 4242))
    app.run(host="0.0.0.0", port=port, debug=True)
