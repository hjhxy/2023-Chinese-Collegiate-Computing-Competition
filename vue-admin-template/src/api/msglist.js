import request from '@/utils/request'

// 用户管理

export function getList(params) {
  return request({
    url: '/system/msglist',
    method: 'get',
    params
  })
}
