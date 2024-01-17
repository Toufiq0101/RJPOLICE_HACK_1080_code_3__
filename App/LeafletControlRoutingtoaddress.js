function fetchVideoFiles(file) {
  fetch(file)
    .then((response) => response.text())
    .then((data) => {
      // Parse the CSV data using PapaParse
      Papa.parse(data, {
        header: true, // Treat the first row as headers
        complete: function (results) {
          // Extract the 'video_files' column
          const videoFiles = results.data.map((row) => row.video_files);

          // Log or use the videoFiles array as needed
          console.log(videoFiles);
        },
        error: function (error) {
          console.error("Error parsing CSV:", error.message);
        },
      });
    })
    .catch((error) => console.error("Error fetching CSV:", error));
}

L.LeafletControlRoutingtoaddress = L.Control.extend({
  initialize: function (options) {
    L.Util.setOptions(this, options);
  },

  onAdd: function (map) {
    console.log("SAbhxasbhxabshbx");
    this._map = map;
    marker_startingpoint = this._marker_startingpoint = L.marker([0, 0]);
    marker_target = this._marker_tarket = L.marker([0, 0]);
    route_linestring = this._route_linestring = [{}];

    if (
      // typeof json_obj_startingpoint[0] === "undefined" ||
      // json_obj_startingpoint[0] === null ||
      // typeof json_obj_target[0] === "undefined" ||
      // json_obj_target[0] === null
      false
    ) {
      console.log("test_2323");
      input.placeholder = this.options.errormessage;
      this._input.value = "";
    } else {
      console.log("test");
      // this._marker_target = L.marker([
      //   json_obj_target[0].lat,
      //   json_obj_target[0].lon,
      // ]).addTo(this._map);
      // this._marker_startingpoint = L.marker([
      //   json_obj_startingpoint[0].lat,
      //   json_obj_startingpoint[0].lon,
      // ]).addTo(this._map);
      // Example usage
      fetchVideoFiles("introgations_results.csv");

      var json_obj_route;
      if (this.options.router === "mapbox") {
        json_obj_route = JSON.parse(
          Get(
            "https://api.mapbox.com/directions/v5/mapbox/driving/" +
              "87.25057780431678" +
              "," +
              "24.265631401899565" +
              ";" +
              "87.25070997595422" +
              "," +
              "24.264092175463635" +
              "?access_token=" +
              this.options.token +
              "&overview=full&geometries=geojson"
          )
        );
      } else {
        json_obj_route = JSON.parse(
          Get(
            "https://router.project-osrm.org/route/v1/driving/" +
              "87.15558101610473" +
              "," +
              "24.21477285697079" +
              ";" +
              "87.25070997595422" +
              "," +
              "24.264092175463635" +
              "?overview=full&geometries=geojson"
          )
        );
        this._route_linestring = L.geoJSON(
          json_obj_route.routes[0].geometry
        ).addTo(this._map);
        console.log(json_obj_route);
        json_obj_route = JSON.parse(
          Get(
            "https://router.project-osrm.org/route/v1/driving/" +
              "87.23502185757555" +
              "," +
              "24.2867836065823" +
              ";" +
              "87.25089479937802" +
              "," +
              "24.263476049251" +
              "?overview=full&geometries=geojson"
          )
        );
        console.log(json_obj_route);
        this._route_linestring = L.geoJSON(
          json_obj_route.routes[0].geometry
        ).addTo(this._map);
        this._map.fitBounds(this._route_linestring.getBounds());
      }
    }

    function Get(url) {
      var Httpreq = new XMLHttpRequest(); // a new request
      Httpreq.open("GET", url, false);
      Httpreq.send(null);
      return Httpreq.responseText;
    }
  },
  onRemove: function (map) {},
});

L.leafletControlRoutingtoaddress = function (options) {
  return new L.LeafletControlRoutingtoaddress(options);
};
