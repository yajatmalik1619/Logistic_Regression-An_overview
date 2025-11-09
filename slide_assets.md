# Logistic Regression — Concept & Example

---

## 🔹 Introduction

**Goal:** Model the probability that an input belongs to a certain class (e.g., benign vs malignant tumor).  
Unlike linear regression, logistic regression predicts probabilities constrained between 0 and 1.

---

## ⚙️ The Logistic Function

The logistic (sigmoid) function converts any real value into a probability:

sigma(z) = e^z/(1+e^z)
OR
σ(z) = e^z/(1+e^z)

where z=w⊤x+b
**Interpretation:**  
As z→+∞ , σ(z)→1
As z→−∞ , σ(z)→0

**Slide image:**  
📊 *Sigmoid Function — `assets/sigmoid.png`*

---

## 🔹 Decision Boundary Intuition

A linear model separates the input space using a decision boundary where \( P(y=1|x)=0.5 \).  

- Samples on one side → predicted 1 (positive class)  
- Samples on the other → predicted 0 (negative class)

**Slide image:**  
🎨 *Decision Boundary — `assets/decision_boundary.png`*

---

## 📐 Log-Loss Function (Cross Entropy)

To train the model, we minimize **log-loss**:

ℓ(w)=−N1​i=1∑N​[yi​log(y^​i​)+(1−yi​)log(1−y^​i​)]

where 
y^​i​=σ(w⊤xi​)=1+e−(w⊤xi​)1​

- Penalizes confident wrong predictions heavily  
- Encourages probabilities to match true labels

**Slide image:**  
📉 *Logistic Loss for y=0 and y=1 — `assets/logistic_loss.png`*

---

## 🧠 Gradient Descent Update Rule

We update model weights by moving opposite to the gradient:

w←w−η∇wL
where η is the learning rate controlling the step size.

The gradient of the loss is:

∇w​L=N1​i=1∑N​(σ(zi​)−yi​)xi​
---

## 🔍 Interpreting Coefficients

Each coefficient wj represents the influence of feature xj on the log-odds:

logP(y=0∣x)P(y=1∣x)​=w⊤x+b

- Positive wj : increases likelihood of class 1  
- Negative wj : decreases likelihood of class 1  

**Slide image:**  
📊 *Top Feature Coefficients — `assets/coefficients.png`*

---

## 🧬 Application Example — Breast Cancer Prediction

Dataset: *Breast Cancer Wisconsin (Diagnostic)*  
- 30 features describing cell nuclei  
- Target: 0 = Malignant, 1 = Benign

Model: Logistic Regression (with StandardScaler & train/test split)  

**Performance Highlights:**
| Metric | Value |
|:--|--:|
| Accuracy | ~97% |
| ROC-AUC | ~0.99 |
| Precision | ~0.97 |
| Recall | ~0.97 |

**Slide image suggestions:**  
- ROC curve (from app)  
- Confusion matrix (from app)

---

## 💡 Key Takeaways

- Logistic regression is **interpretable** and **efficient** for binary classification.  
- Outputs **probabilities**, not just class labels.  
- Great baseline model before trying more complex algorithms.  
- Coefficients reveal **feature importance** in decision making.

---

## 📚 Recommended Reading

- *Scikit-learn Documentation* — Logistic Regression User Guide  
- *Andrew Ng* — Machine Learning Lecture Notes (Coursera)  
- *The Elements of Statistical Learning* — Chapter on Generalized Linear Models  
- *Sebastian Raschka* — Logistic Regression Explained (blog post)  

---

## 🏁 Conclusion

The logistic regression model provides a strong, interpretable foundation for medical diagnosis problems.  
By mapping continuous input data into probability space using the sigmoid function,  
it effectively distinguishes between malignant and benign tumors.

