output "ec2_spot_instance_id" {
  value = aws_spot_instance_request.training.spot_instance_id
}

output "ec2_public_ip" {
  value = aws_spot_instance_request.training.public_ip
}
