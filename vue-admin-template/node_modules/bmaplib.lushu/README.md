# BMapLib.LuShu

## Installation

### NPM

```bash
$ npm i --save bmaplib.lushu
```

### CDN

```html
<script src="//unpkg.com/bmaplib.lushu"></script>
```

## Usage

### ES Next

```js
import LuShu from 'bmaplib.lushu'

// You should use this lib after BaiduMap loaded. For Example:

loadBaiduMap.then(() => {
  new LuShu()
})
```

### CDN

```html
<script src="//api.map.baidu.com/api?v=2.0"></script>
<script src="//unpkg.com/bmaplib.lushu"></script>
<script>
  new BMapLib.Lushu()
</script>
```