(function (root, factory) {  
    if (typeof exports === 'object') {  
        module.exports = factory();
    } else if (typeof define === 'function' && define.amd) {  
        define(factory);  
    } else {
        root.BMapLib = root.BMapLib || {};
        root.BMapLib.LuShu = root.BMapLib.Lushu || factory();  
    }  
})(this, function() {
    var baidu = {};
    baidu.dom = {};
    baidu.dom.g = function(id) {
        if ('string' == typeof id || id instanceof String) {
            return document.getElementById(id);
        } else if (id && id.nodeName && (id.nodeType == 1 || id.nodeType == 9)) {
            return id;
        }
        return null;
    };
    baidu.g = baidu.G = baidu.dom.g;
    baidu.lang = baidu.lang || {};
    baidu.lang.isString = function(source) {
        return '[object String]' == Object.prototype.toString.call(source);
    };
    baidu.isString = baidu.lang.isString;
    baidu.dom._g = function(id) {
        if (baidu.lang.isString(id)) {
            return document.getElementById(id);
        }
        return id;
    };
    baidu._g = baidu.dom._g;
    baidu.dom.getDocument = function(element) {
        element = baidu.dom.g(element);
        return element.nodeType == 9 ? element : element.ownerDocument || element.document;
    };
    baidu.browser = baidu.browser || {};
    baidu.browser.ie = baidu.ie = /msie (\d+\.\d+)/i.test(navigator.userAgent) ? (document.documentMode || + RegExp['\x241']) : undefined;
    baidu.dom.getComputedStyle = function(element, key) {
        element = baidu.dom._g(element);
        var doc = baidu.dom.getDocument(element),
            styles;
        if (doc.defaultView && doc.defaultView.getComputedStyle) {
            styles = doc.defaultView.getComputedStyle(element, null);
            if (styles) {
                return styles[key] || styles.getPropertyValue(key);
            }
        }
        return '';
    };
    baidu.dom._styleFixer = baidu.dom._styleFixer || {};
    baidu.dom._styleFilter = baidu.dom._styleFilter || [];
    baidu.dom._styleFilter.filter = function(key, value, method) {
        for (var i = 0, filters = baidu.dom._styleFilter, filter; filter = filters[i]; i++) {
            if (filter = filter[method]) {
                value = filter(key, value);
            }
        }
        return value;
    };
    baidu.string = baidu.string || {};
    baidu.string.toCamelCase = function(source) {

        if (source.indexOf('-') < 0 && source.indexOf('_') < 0) {
            return source;
        }
        return source.replace(/[-_][^-_]/g, function(match) {
            return match.charAt(1).toUpperCase();
        });
    };
    baidu.dom.getStyle = function(element, key) {
        var dom = baidu.dom;
        element = dom.g(element);
        key = baidu.string.toCamelCase(key);

        var value = element.style[key] ||
                    (element.currentStyle ? element.currentStyle[key] : '') ||
                    dom.getComputedStyle(element, key);

        if (!value) {
            var fixer = dom._styleFixer[key];
            if (fixer) {
                value = fixer.get ? fixer.get(element) : baidu.dom.getStyle(element, fixer);
            }
        }

        if (fixer = dom._styleFilter) {
            value = fixer.filter(key, value, 'get');
        }
        return value;
    };
    baidu.getStyle = baidu.dom.getStyle;
    baidu.dom._NAME_ATTRS = (function() {
        var result = {
            'cellpadding': 'cellPadding',
            'cellspacing': 'cellSpacing',
            'colspan': 'colSpan',
            'rowspan': 'rowSpan',
            'valign': 'vAlign',
            'usemap': 'useMap',
            'frameborder': 'frameBorder'
        };

        if (baidu.browser.ie < 8) {
            result['for'] = 'htmlFor';
            result['class'] = 'className';
        } else {
            result['htmlFor'] = 'for';
            result['className'] = 'class';
        }

        return result;
    })();
    baidu.dom.setAttr = function(element, key, value) {
        element = baidu.dom.g(element);
        if ('style' == key) {
            element.style.cssText = value;
        } else {
            key = baidu.dom._NAME_ATTRS[key] || key;
            element.setAttribute(key, value);
        }
        return element;
    };
    baidu.setAttr = baidu.dom.setAttr;
    baidu.dom.setAttrs = function(element, attributes) {
        element = baidu.dom.g(element);
        for (var key in attributes) {
            baidu.dom.setAttr(element, key, attributes[key]);
        }
        return element;
    };
    baidu.setAttrs = baidu.dom.setAttrs;
    baidu.dom.create = function(tagName, opt_attributes) {
        var el = document.createElement(tagName),
            attributes = opt_attributes || {};
        return baidu.dom.setAttrs(el, attributes);
    };
    baidu.object = baidu.object || {};
    baidu.extend =
    baidu.object.extend = function(target, source) {
        for (var p in source) {
            if (source.hasOwnProperty(p)) {
                target[p] = source[p];
            }
        }
        return target;
    };

    /**
     * @exports LuShu as BMapLib.LuShu
     */
    var LuShu = function(map, path, opts) {
        try {
            BMap;
        } catch (e) {
            throw Error('Baidu Map JS API is not ready yet!');
        }
        if (!path || path.length < 1) {
            return;
        }
        this._map = map;
        this._path = path;
        this.i = 0;
        this._setTimeoutQuene = [];
        this._projection = this._map.getMapType().getProjection();
        this._opts = {
            icon: null,
            speed: 4000,
            defaultContent: '',
            showInfoWindow: false
        };
        this._setOptions(opts);
        this._rotation = 0;

        if (!this._opts.icon instanceof BMap.Icon) {
            this._opts.icon = defaultIcon;
        }
    }
    LuShu.prototype._setOptions = function(opts) {
        if (!opts) {
            return;
        }
        for (var p in opts) {
            if (opts.hasOwnProperty(p)) {
                this._opts[p] = opts[p];
            }
        }
    }
    LuShu.prototype.start = function() {
        var me = this,
            len = me._path.length;
        this._opts.onstart && this._opts.onstart(me)
        if (me.i && me.i < len - 1) {
            if (!me._fromPause) {
                return;
            }else if(!me._fromStop){
	            me._moveNext(++me.i);
            }
        }else {
            !me._marker && me._addMarker();
            me._timeoutFlag = setTimeout(function() {
                    !me._overlay && me._addInfoWin();
                    me._moveNext(me.i);
            },400);
        }
        this._fromPause = false;
        this._fromStop = false;
    },
    LuShu.prototype.stop = function() {
        this.i = 0;
        this._fromStop = true;
        clearInterval(this._intervalFlag);
        this._clearTimeout();
        for (var i = 0, t = this._opts.landmarkPois, len = t.length; i < len; i++) {
            t[i].bShow = false;
        }
        this._opts.onstop && this._opts.onstop(this)
    };
    LuShu.prototype.pause = function() {
        clearInterval(this._intervalFlag);
        this._fromPause = true;
        this._clearTimeout();
        this._opts.onpause && this._opts.onpause(this)
    };
    LuShu.prototype.hideInfoWindow = function() {
        this._opts.showInfoWindow = false;
        this._overlay && (this._overlay._div.style.visibility = 'hidden');
    };
    LuShu.prototype.showInfoWindow = function() {
        this._opts.showInfoWindow = true;
        this._overlay && (this._overlay._div.style.visibility = 'visible');
    };
    LuShu.prototype.dispose = function () {
        clearInterval(this._intervalFlag);
        this._setTimeoutQuene && this._clearTimeout();
        if (this._map) {
            this._map.removeOverlay(this._overlay);
            this._map.removeOverlay(this._marker);
        }
    };
    baidu.object.extend(LuShu.prototype, {
        _addMarker: function(callback) {
            if (this._marker) {
                this.stop();
                this._map.removeOverlay(this._marker);
                clearTimeout(this._timeoutFlag);
            }
            this._overlay && this._map.removeOverlay(this._overlay);
            var marker = new BMap.Marker(this._path[0]);
            this._opts.icon && marker.setIcon(this._opts.icon);
            this._map.addOverlay(marker);
            marker.setAnimation(BMAP_ANIMATION_DROP);
            this._marker = marker;
        },
        _addInfoWin: function() {
            var me = this;
            !CustomOverlay.prototype.initialize && initCustomOverlay();
            var overlay = new CustomOverlay(me._marker.getPosition(), me._opts.defaultContent);
            overlay.setRelatedClass(this);
            this._overlay = overlay;
            this._map.addOverlay(overlay);
            this._opts.showInfoWindow ? this.showInfoWindow() : this.hideInfoWindow()
        },
        _getMercator: function(poi) {
            return this._map.getMapType().getProjection().lngLatToPoint(poi);
        },
        _getDistance: function(pxA, pxB) {
            return Math.sqrt(Math.pow(pxA.x - pxB.x, 2) + Math.pow(pxA.y - pxB.y, 2));
        },
        _move: function(initPos,targetPos,effect) {
            var me = this,
                currentCount = 0,
                timer = 10,
                step = this._opts.speed / (1000 / timer),
                init_pos = this._projection.lngLatToPoint(initPos),
                target_pos = this._projection.lngLatToPoint(targetPos),
                count = Math.round(me._getDistance(init_pos, target_pos) / step);
            if (count < 1) {
                me._moveNext(++me.i);
                return;
            }
            me._intervalFlag = setInterval(function() {
	            if (currentCount >= count) {
	                clearInterval(me._intervalFlag);
		        	if(me.i > me._path.length){
						return;
		        	}
	                me._moveNext(++me.i);
	            } else {
                    currentCount++;
                    var x = effect(init_pos.x, target_pos.x, currentCount, count),
                        y = effect(init_pos.y, target_pos.y, currentCount, count),
                        pos = me._projection.pointToLngLat(new BMap.Pixel(x, y));
                    if(currentCount == 1){
                        var proPos = null;
                        if(me.i - 1 >= 0){
                            proPos = me._path[me.i - 1];
                        }
                        if(me._opts.enableRotation == true){
                            me.setRotation(proPos,initPos,targetPos);
                        }
                        if(me._opts.autoView){
                            if(!me._map.getBounds().containsPoint(pos)){
                                me._map.setCenter(pos);
                            }   
                        }
                    }
                    me._marker.setPosition(pos);
                    me._setInfoWin(pos);
                }
	        },timer);
        },
        setRotation : function(prePos,curPos,targetPos){
            var me = this;
            var deg = 0;
            //start!
            curPos =  me._map.pointToPixel(curPos);
            targetPos =  me._map.pointToPixel(targetPos);   

            if(targetPos.x != curPos.x){
                var tan = (targetPos.y - curPos.y)/(targetPos.x - curPos.x),
                atan  = Math.atan(tan);
                deg = atan*360/(2*Math.PI);
                //degree  correction;
                if(targetPos.x < curPos.x){
                    deg = -deg + 90 + 90;

                } else {
                    deg = -deg;
                }

                me._marker.setRotation(-deg);   

            }else {
                var disy = targetPos.y- curPos.y ;
                var bias = 0;
                if(disy > 0)
                    bias=-1
                else
                    bias = 1
                me._marker.setRotation(-bias * 90);  
            }
            return;
        },
        linePixellength : function(from,to){ 
            return Math.sqrt(Math.abs(from.x- to.x) * Math.abs(from.x- to.x) + Math.abs(from.y- to.y) * Math.abs(from.y- to.y) );
        },
        pointToPoint : function(from,to ){
            return Math.abs(from.x- to.x) * Math.abs(from.x- to.x) + Math.abs(from.y- to.y) * Math.abs(from.y- to.y)
        },
        _moveNext: function(index) {
            var me = this;
            if (index < this._path.length - 1) {
                me._move(me._path[index], me._path[index + 1], me._tween.linear);
            } else {
                me.stop()
            }
        },
        _setInfoWin: function(pos) {
            var me = this;
            me._overlay.setPosition(pos, me._marker.getIcon().size);
            var index = me._troughPointIndex(pos);
            if (index != -1) {
                clearInterval(me._intervalFlag);
                me._overlay.setHtml(me._opts.landmarkPois[index].html);
                me._overlay.setPosition(pos, me._marker.getIcon().size);
                me._pauseForView(index);
            }else {
                me._overlay.setHtml(me._opts.defaultContent);
            }
        },
        _pauseForView: function(index) {
            var me = this;
            var t = setTimeout(function() {
                me._moveNext(++me.i);
            },me._opts.landmarkPois[index].pauseTime * 1000);
            me._setTimeoutQuene.push(t);
        },
        _clearTimeout: function() {
            for (var i in this._setTimeoutQuene) {
                clearTimeout(this._setTimeoutQuene[i]);
            }
            this._setTimeoutQuene.length = 0;
        },
        _tween: {
            linear: function(initPos, targetPos, currentCount, count) {
                var b = initPos, c = targetPos - initPos, t = currentCount,
                d = count;
                return c * t / d + b;
            }
        },
        _troughPointIndex: function(markerPoi) {
            var t = this._opts.landmarkPois, distance;
            for (var i = 0, len = t.length; i < len; i++) {
                if (!t[i].bShow) {
                    distance = this._map.getDistance(new BMap.Point(t[i].lng, t[i].lat), markerPoi);
                    if (distance < 10) {
                        t[i].bShow = true;
                        return i;
                    }
                }
            }
           return -1;
        }
    });
    function CustomOverlay(point,html) {
        this._point = point;
        this._html = html;
    }

    function initCustomOverlay() {
        CustomOverlay.prototype = new BMap.Overlay();
        CustomOverlay.prototype.initialize = function(map) {
            var div = this._div = baidu.dom.create('div', {style: 'border:solid 1px #ccc;width:auto;min-width:50px;text-align:center;position:absolute;background:#fff;color:#000;font-size:12px;border-radius: 10px;padding:5px;white-space: nowrap;'});
            div.innerHTML = this._html;
            map.getPanes().floatPane.appendChild(div);
            this._map = map;
            return div;
        }
        CustomOverlay.prototype.draw = function() {
            this.setPosition(this.lushuMain._marker.getPosition(), this.lushuMain._marker.getIcon().size);
        }
        baidu.object.extend(CustomOverlay.prototype, {
            setPosition: function(poi,markerSize) {
                var px = this._map.pointToOverlayPixel(poi),
                    styleW = baidu.dom.getStyle(this._div, 'width'),
                    styleH = baidu.dom.getStyle(this._div, 'height'),
                    overlayW = parseInt(this._div.clientWidth || styleW, 10),
                    overlayH = parseInt(this._div.clientHeight || styleH, 10);
                this._div.style.left = px.x - overlayW / 2 + 'px';
                this._div.style.bottom = -(px.y - markerSize.height) + 'px';
            },
            setHtml: function(html) {
                this._div.innerHTML = html;
            },
            setRelatedClass: function(lushuMain) {
                this.lushuMain = lushuMain;
            }
        });
    }

    return LuShu
});
