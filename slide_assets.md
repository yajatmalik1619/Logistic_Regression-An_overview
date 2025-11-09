# Logistic Regression — Concept & Example

---

## 🔹 Introduction

**Goal:** Model the probability that an input belongs to a certain class (e.g., benign vs malignant tumor).  
Unlike linear regression, logistic regression predicts probabilities constrained between 0 and 1.

---

## ⚙️ The Logistic Function

The logistic (sigmoid) function converts any real value into a probability:

sigma(z) = e^z/(1+e^z)

where \( z = \mathbf{w}^\top \mathbf{x} + b \)

**Interpretation:**  
- As \( z \to +\infty \), \( \sigma(z) \to 1 \)  
- As \( z \to -\infty \), \( \sigma(z) \to 0 \)

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

\[
\ell(\mathbf{w}) = - \frac{1}{N}\sum_{i=1}^N 
\big[y_i \log(\hat{y_i}) + (1 - y_i)\log(1 - \hat{y_i})\big]
\]
where \( \hat{y_i} = \sigma(\mathbf{w}^\top \mathbf{x_i}) \)

- Penalizes confident wrong predictions heavily  
- Encourages probabilities to match true labels

**Slide image:**  
📉 *Logistic Loss for y=0 and y=1 — `assets/logistic_loss.png`*

---

## 🧠 Gradient Descent Update Rule

We update model weights by moving opposite to the gradient:

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla_{\mathbf w} L
\]
where \( \eta \) is the learning rate.

The gradient of the loss is:

\[
\nabla_{\mathbf w} L = \frac{1}{N} \sum_{i=1}^N (\sigma(z_i) - y_i)\mathbf{x_i}
\]

---

## 🔍 Interpreting Coefficients

Each coefficient \( w_j \) represents the influence of feature \( x_j \) on the log-odds:

\[
\log\frac{P(y=1\mid\mathbf{x})}{P(y=0\mid\mathbf{x})} = \mathbf{w}^\top \mathbf{x} + b
\]

- Positive \( w_j \): increases likelihood of class 1  
- Negative \( w_j \): decreases likelihood of class 1  

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

