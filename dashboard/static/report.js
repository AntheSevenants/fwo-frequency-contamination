// 1. Use Papa Parse to fetch and parse the URL
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
            autoColumns: true
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