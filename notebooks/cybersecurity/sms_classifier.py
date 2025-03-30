from typing import Literal

import dspy

class SMSClassifierSignature(dspy.Signature):
    """
    Given an SMS text, predict whether it is ham, spam, or smishing.
    Output only the predicted label.
    """

    sms_text: str = dspy.InputField()
    category : Literal["ham", "spam", "smishing"] = dspy.OutputField()


class SMSClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(
            SMSClassifierSignature
        )

    def forward(self, sms_text):
        return self.generate_answer(sms_text=sms_text)
