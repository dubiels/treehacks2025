async function solveCaptcha() {
    const input = document.getElementById("captchaInput");
    if (!input.files.length) {
        alert("Please upload a CAPTCHA image.");
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append("image", file);

    document.getElementById("correctResponse").innerText = "Processing...";

    try {
        const response = await fetch("/solve", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        document.getElementById("correctResponse").innerText = data.correct_response || "N/A";

        const table = document.getElementById("resultsTable");
        table.innerHTML = "";  // Clear previous results

        data.results.forEach((res) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td class="py-2">${res.agent}</td>
                <td class="py-2 ${res.correct ? 'text-green-400' : 'text-red-400'}">${res.response}</td>
                <td class="py-2">${res.time}s</td>
            `;
            table.appendChild(row);
        });
    } catch (error) {
        console.error("Error solving CAPTCHA:", error);
        document.getElementById("correctResponse").innerText = "Error.";
    }
}
