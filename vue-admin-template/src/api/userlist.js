import request from '@/utils/request'

// 信息审核

export function getList(params) {
  return request({
    url: '/system/userlist',
    method: 'get',
    params
  })
}
// '/user/deleteuser'
export function deleteUser(username) {
  return request({
    url: '/system/userlist',
    method: 'POST',
    data: {
      username
    }
  })
}
