# BMapLib.Heatmap

## Installation

### NPM

```bash
$ npm i --save bmaplib.heatmap
```

### CDN

```html
<script src="//unpkg.com/bmaplib.heatmap"></script>
```

## Usage

### ES Next

```js
import Heatmap from 'bmaplib.heatmap'

// You should use this lib after BaiduMap loaded. For Example:

loadBaiduMap.then(() => {
  new Heatmap()
})
```

### CDN

```html
<script src="//api.map.baidu.com/api?v=2.0"></script>
<script src="//unpkg.com/bmaplib.heatmap"></script>
<script>
  new BMapLib.Heatmap()
</script>
```
