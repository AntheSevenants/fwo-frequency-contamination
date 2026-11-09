function formatter(cell) {
    const value = cell.getValue();
    console.log(typeof value);
    if (typeof value !== "string") {
        return;
    }

    if (value == "contamination") {
        cell.getElement().classList.add("cell-red");
    } else if (value == "avoidance") {
        cell.getElement().classList.add("cell-green");
    } else {
        cell.getElement().classList.add("cell-yellow");
    }

    return value;
}

function generalFormatter(cell) {
    const value = cell.getValue();

    if (typeof value === "number" && !Number.isNaN(value)) {
        return Math.round(value * 100) / 100;
    }

    return value;
}

Papa.parse(reportUrl, {
    download: true,      // This tells Papa Parse to fetch the URL
    header: true,        // Converts rows into objects using the first row as keys
    dynamicTyping: true, // Automatically converts numbers and booleans (very helpful!)
    skipEmptyLines: true,
    importFormat: "csv",
    complete: function (results) {
        // 2. Once parsing is finished, initialize Tabulator with the data
        let table = new Tabulator("#report_table", {
            data: results.data, // The array of objects from the CSV
            layout: "fitColumns",
            // pagination: "local",
            // paginationSize: 10,
            autoColumns: true,
            autoColumnsDefinitions: (definitions) => {
                definitions.forEach((column) => {
                    // Add header filter to every column
                    column.headerFilter = true;

                    // Apply your custom formatter to specific fields
                    if (column.field === "expectation" || column.field === "outcome") {
                        column.formatter = formatter;
                    } else {
                        column.formatter = generalFormatter;
                    }
                });

                return definitions;
            },
        });

        table.on("rowClick", function (e, row) {
            const data = row.getData();
            console.log(clickThroughUrl + data.combination_id);
            window.location.href = clickThroughUrl + data.combination_id;
        });
    },
    error: function (err) {
        console.error("Error loading CSV:", err);
    }
});