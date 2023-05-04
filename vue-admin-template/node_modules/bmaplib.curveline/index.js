(function (root, factory) {  
  if (typeof exports === 'object') {  
      module.exports = factory();
  } else if (typeof define === 'function' && define.amd) {  
      define(factory);  
  } else {
      root.BMapLib = root.BMapLib || {};
      root.BMapLib.CurveLine = root.BMapLib.CurveLine || factory();  
  }  
})(this, function() {
  function CurveLine(points, opts) {
    try {
      BMap;
    } catch (e) {
      throw Error('Baidu Map JS API is not ready yet!');
    }
    var self = this;
    var curvePoints = getCurvePoints(points);
    var polyline = new BMap.Polyline(curvePoints, opts);

    polyline.addEventListener('lineupdate', function(){
      if (this.isEditing) {
        this.enableEditing();
      }
    });

    polyline.cornerPoints = points;
    polyline.editMarkers = []; //缂栬緫鍔熻兘鐨勯《鐐�

    /**
     * 閲嶅啓寮х嚎鐨勭紪杈戝姛鑳�
     */
    polyline.enableEditing = function () {
      var self = this;

      if (self.map) {
        self.disableEditing();
        for (var i = 0; i < self.cornerPoints.length; i++) {
          var marker = new BMap.Marker(self.cornerPoints[i], {
            icon: new BMap.Icon('http://api.map.baidu.com/library/CurveLine/1.5/src/circle.png', new BMap.Size(16,16)),
            enableDragging: true,
            raiseOnDrag: true
          });
          marker.addEventListener('dragend', function(){
            self.cornerPoints.length = 0;
            for (var i = 0; i < self.editMarkers.length; i++) {
              self.cornerPoints.push(self.editMarkers[i].getPosition());
            }
            var curvePoints = getCurvePoints(self.cornerPoints)
            self.setPath(curvePoints);
          });
          marker.index = i;
          self.editMarkers.push(marker);
          self.map.addOverlay(marker);
        }
      }
      self.isEditing = true;
    }

    /**
     * 閲嶅啓寮х嚎鐨勭紪杈戝姛鑳�
     */
    polyline.disableEditing = function() {
      this.isEditing = false;
      //娓呯┖涔嬪墠鐨勭紪杈戠偣
      for (var i = 0; i < this.editMarkers.length; i++) {
        this.map.removeOverlay(this.editMarkers[i]);
        this.editMarkers[i] = null;
      }
      this.editMarkers.length = 0;
    }

    /**
     * 鑾峰彇寮х嚎鐨勫潗鏍囩偣
     */
    polyline.getPath = function() {
      return curvePoints;
    }

    // polyline.points = points; //寮х嚎鐨勫潗鏍囬《鐐�
    // polyline.enableEditing = enableEditing;
    // polyline.disableEditing = disableEditing;
    // polyline.getPath = getPath;
    return polyline;
  }

  function extend(child, parent) {
    for (var p in parent) {
      if (parent.hasOwnProperty(p)) {
        child[p] = parent[p];
      }
    }
    
    return child;
  };

  function getCurvePoints(points) {
    var curvePoints = [];
    for (var i = 0; i < points.length - 1; i++) {
      var p = getCurveByTwoPoints(points[i], points[i + 1]);
      if (p && p.length > 0) {
        curvePoints = curvePoints.concat(p);
      }
    }
    return curvePoints;
  }

  function getCurveByTwoPoints(obj1, obj2) {
    var curveCoordinates = [];
    if (!obj1 || !obj2 || !(obj1 instanceof BMap.Point) || !(obj2 instanceof BMap.Point)) {
      return null;
    }

    var B1 = function(x) {
      return 1 - 2 * x + x * x;
    };
    var B2 = function(x) {
      return 2 * x - 2 * x * x;
    };
    var B3 = function(x) {
      return x * x;
    };

    var count=30; // 鏇茬嚎鏄敱涓€浜涘皬鐨勭嚎娈电粍鎴愮殑锛岃繖涓〃绀鸿繖涓洸绾挎墍鏈夊埌鐨勬姌绾跨殑涓暟
    var isFuture=false;
    var t, h, h2, lat3, lng3, j, t2;
    var LnArray = [];
    var i = 0;
    var inc = 0;

    if (typeof(obj2) == "undefined") {
      if (typeof(curveCoordinates) != "undefined") {
        curveCoordinates = [];
      }
      return;
    }

    var lat1 = parseFloat(obj1.lat);
    var lat2 = parseFloat(obj2.lat);
    var lng1 = parseFloat(obj1.lng);
    var lng2 = parseFloat(obj2.lng);
      
    // 璁＄畻鏇茬嚎瑙掑害鐨勬柟娉�
    if (lng2 > lng1) {
      if (parseFloat(lng2-lng1) > 180) {
        if (lng1 < 0) {
          lng1 = parseFloat(180 + 180 + lng1);
        }
      }
    }
    
    if (lng1 > lng2) {
      if (parseFloat(lng1-lng2) > 180) {
        if (lng2 < 0) {
          lng2 = parseFloat(180 + 180 + lng2);
        }
      }
    }
    j = 0;
    t2 = 0;
    if (lat2 == lat1) {
      t = 0;
      h = lng1 - lng2;
    } else if (lng2 == lng1) {
      t = Math.PI / 2;
      h = lat1 - lat2;
    } else {
      t = Math.atan((lat2 - lat1) / (lng2 - lng1));
      h = (lat2 - lat1) / Math.sin(t);
    }
    if (t2 == 0) {
      t2 = (t + (Math.PI / 5));
    }
    h2 = h / 2;
    lng3 = h2 * Math.cos(t2) + lng1;
    lat3 = h2 * Math.sin(t2) + lat1;

    for (i = 0; i < count + 1; i++) {
      curveCoordinates.push(new BMap.Point(
        (lng1 * B1(inc) + lng3 * B2(inc)) + lng2 * B3(inc),
        (lat1 * B1(inc) + lat3 * B2(inc) + lat2 * B3(inc))
      ));
      inc = inc + (1 / count);
    }
    return curveCoordinates;
  }
  return CurveLine;
});
