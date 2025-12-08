import stripe
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

YOUR_DOMAIN = os.getenv("APP_DOMAIN", "http://localhost:8501")


@app.post("/create-checkout-session")
def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": 500,  # $5.00 example
                        "product_data": {
                            "name": "Waste Classifier Pro",
                        },
                    },
                    "quantity": 1,
                }
            ],
            mode="payment",
            success_url=f"{YOUR_DOMAIN}/success",
            cancel_url=f"{YOUR_DOMAIN}/cancel",
        )
        return jsonify({"id": session.id})

    except Exception as e:
        return jsonify(error=str(e)), 403


if __name__ == "__main__":
    app.run(port=4242, debug=True)
