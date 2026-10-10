/* ==================================================
   THEME
================================================== */

const themeToggle = document.getElementById("theme-toggle");

const savedTheme = localStorage.getItem("theme");

if (savedTheme === "light") {
    document.body.classList.add("light");
}

themeToggle.addEventListener("click", () => {

    document.body.classList.toggle("light");

    const theme =
        document.body.classList.contains("light")
            ? "light"
            : "dark";

    localStorage.setItem("theme", theme);

});


/* ==================================================
   SCROLL REVEAL
================================================== */

const revealElements =
    document.querySelectorAll(".reveal");

const observer =
    new IntersectionObserver(
        (entries) => {

            entries.forEach((entry) => {

                if (entry.isIntersecting) {

                    entry.target.classList.add("visible");

                    observer.unobserve(entry.target);

                }

            });

        },
        {
            threshold: 0.12
        }
    );


revealElements.forEach((element) => {

    observer.observe(element);

});


/* ==================================================
   PARALLAX HERO
================================================== */

const heroVisual =
    document.querySelector(".hero-visual");

window.addEventListener("mousemove", (event) => {

    if (!heroVisual) return;

    const x =
        (event.clientX / window.innerWidth - 0.5) * 8;

    const y =
        (event.clientY / window.innerHeight - 0.5) * 8;

    heroVisual.style.transform =
        `translate(${x}px, ${y}px)`;

});


/* ==================================================
   PREDICTION FORM
================================================== */

const form =
    document.getElementById("prediction-form");

const result =
    document.getElementById("result");


form.addEventListener("submit", async (event) => {

    event.preventDefault();


    const data = {

        age:
            Number(
                document.getElementById("age").value
            ),

        job:
            document.getElementById("job").value,

        marital:
            document.getElementById("marital").value,

        education:
            document.getElementById("education").value,

        default:
            document.getElementById("default").value,

        balance:
            Number(
                document.getElementById("balance").value
            ),

        housing:
            document.getElementById("housing").value,

        loan:
            document.getElementById("loan").value,

        contact:
            document.getElementById("contact").value,

        day:
            Number(
                document.getElementById("day").value
            ),

        month:
            document.getElementById("month").value,

        campaign:
            Number(
                document.getElementById("campaign").value
            ),

        pdays:
            Number(
                document.getElementById("pdays").value
            ),

        previous:
            Number(
                document.getElementById("previous").value
            ),

        poutcome:
            document.getElementById("poutcome").value

    };


    result.innerHTML = `
        <div class="empty-result">

            <div class="empty-icon">
                ◌
            </div>

            <h3>Đang phân tích...</h3>

            <p>
                Mô hình đang xử lý dữ liệu khách hàng.
            </p>

        </div>
    `;


    try {

        const response =
            await fetch("/api/score", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify(data)

            });


        const dataResult =
            await response.json();


        if (!response.ok) {

            throw new Error(
                dataResult.error ||
                "Có lỗi xảy ra."
            );

        }


        const probability =
            (
                dataResult.probability * 100
            ).toFixed(2);


        const isPositive =
            dataResult.prediction === 1;


        const decision =
            isPositive
                ? "Có khả năng đăng ký"
                : "Không có khả năng đăng ký";


        const decisionClass =
            isPositive
                ? "yes"
                : "no";


        result.innerHTML = `

            <div class="result-box">

                <span class="result-label">
                    XÁC SUẤT DỰ ĐOÁN
                </span>

                <div class="result-probability">
                    ${probability}%
                </div>

                <div class="result-threshold">
                    Threshold:
                    ${dataResult.threshold}
                </div>

                <div
                    class="result-decision ${decisionClass}"
                >
                    ${decision}
                </div>

            </div>

        `;

    }


    catch (error) {

        result.innerHTML = `

            <div class="empty-result">

                <div class="empty-icon">
                    !
                </div>

                <h3>Có lỗi xảy ra</h3>

                <p>
                    ${error.message}
                </p>

            </div>

        `;

    }

});