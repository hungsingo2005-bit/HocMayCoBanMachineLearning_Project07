const form = document.getElementById("prediction-form");
const result = document.getElementById("result");

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const data = {
        age: Number(document.getElementById("age").value),
        job: document.getElementById("job").value,
        marital: document.getElementById("marital").value,
        education: document.getElementById("education").value,
        default: document.getElementById("default").value,
        balance: Number(document.getElementById("balance").value),
        housing: document.getElementById("housing").value,
        loan: document.getElementById("loan").value,
        contact: document.getElementById("contact").value,
        day: Number(document.getElementById("day").value),
        month: document.getElementById("month").value,
        campaign: Number(document.getElementById("campaign").value),
        pdays: Number(document.getElementById("pdays").value),
        previous: Number(document.getElementById("previous").value),
        poutcome: document.getElementById("poutcome").value
    };

    result.innerHTML = "Đang dự đoán...";

    try {

        const response = await fetch("/api/score", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const dataResult = await response.json();

        if (!response.ok) {
            throw new Error(dataResult.error);
        }

        const probability =
            (dataResult.probability * 100).toFixed(2);

        const decision =
            dataResult.decision === "yes"
                ? "YES - Có khả năng đăng ký"
                : "NO - Không có khả năng đăng ký";

        const decisionClass =
            dataResult.prediction === 1
                ? "yes"
                : "no";

        result.innerHTML = `
            <div class="result-box">

                <p>Probability</p>

                <div class="probability">
                    ${probability}%
                </div>

                <p>Threshold: ${dataResult.threshold}</p>

                <div class="decision ${decisionClass}">
                    ${decision}
                </div>

            </div>
        `;

    } catch (error) {

        result.innerHTML = `
            <div class="error">
                ${error.message}
            </div>
        `;
    }
});