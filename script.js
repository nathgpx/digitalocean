document.getElementById("medicalForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    // Coletar dados do formulário
    const name = document.getElementById("name").value;
    const age = document.getElementById("age").value;
    const gender = document.getElementById("gender").value;
    const contact = document.getElementById("contact").value;
    const symptoms = document.getElementById("symptoms").value.split(",").map(symptom => symptom.trim());
    const observations = document.getElementById("observations").value;

    const response = await fetch("/diagnose", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            name: name,
            age: age,
            gender: gender,
            contact: contact,
            symptoms: symptoms,
            observations: observations
        })
    });

    const result = await response.json();

    if (result.error) {
        document.getElementById("result").innerHTML = `<p style="color: red;">${result.error}</p>`;
    } else {
        document.getElementById("result").innerHTML = `
            <h2>Resultado do Diagnóstico</h2>
            <p><strong>Doença prevista:</strong> ${result.predicted_disease}</p>
            <p><strong>Confiança:</strong> ${result.confidence}%</p>
        `;
    }

    document.getElementById("result").style.display = "block";
});
