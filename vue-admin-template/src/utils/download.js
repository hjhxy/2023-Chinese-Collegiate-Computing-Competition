/* 图片的下载*/
export function downloadPic(url, filename) {
  const imgsrc = url
  const image = new Image()
  // 解决跨域canvas污染问题
  image.setAttribute('crossOrigin', 'anonymous')
  image.onload = function() {
    const canvas = document.createElement('canvas')
    canvas.width = image.width
    canvas.height = image.height
    const context = canvas.getContext('2d')
    context.drawImage(image, 0, 0, image.width, image.height)
    const url = canvas.toDataURL('image/png') // 得到图片的base64编码数据
    const a = document.createElement('a')
    a.download = filename
    a.href = url
    a.click()
  }
  image.src = imgsrc
}
/* 视频的下载*/
export function downVideo(url) {
  const xhr = new XMLHttpRequest()
  xhr.open('GET', url, true)
  xhr.addEventListener('progress', function(obj) {
    if (obj.lengthComputable) {
      const percentComplete = obj.loaded / obj.total
      console.log((percentComplete * 100).toFixed(2) + '%')
      // 可得到下载进度
    }
  }, false)
  xhr.responseType = 'blob' // 设置返回类型blob
  xhr.onload = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      const blob = this.response
      // 转换一个blob链接
      const u = window.URL.createObjectURL(new Blob([blob], {
        type: 'video/mp4'
      }))
      const a = document.createElement('a')
      a.download = url // 这里是文件名称，这里暂时用链接代替，可以替换
      a.href = u
      a.style.display = 'none'
      document.body.appendChild(a)
      a.click()
      a.remove()
    }
  }
  xhr.send()
}
