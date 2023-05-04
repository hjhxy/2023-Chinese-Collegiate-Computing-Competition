import request from '@/utils/request'

// 检测历史

export function getList(params) {
  return request({
    url: '/system/checklist',
    method: 'get',
    params
  })
}
